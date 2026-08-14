"""train_unlearn_salun.py

SalUn (Saliency-based Unlearning, ICLR 2024 Spotlight) for the multi-speaker
VITS model on VCTK.

SalUn = weight-saliency mask + random-label fine-tuning on the forget set,
restricted to the salient weights and run at a very low learning rate.

Pipeline
--------
Phase 1 -- weight saliency mask
    On the forget speaker's *randomly-labelled* training data (audio relabelled
    with a random other speaker's sid), compute the gradient magnitude of the
    generator loss:
        saliency = |d L(θ; D_f) / dθ|
    and keep the top `--sparsity` fraction of weights -> a binary mask m.
    (SalUn: m = 1(|grad| >= gamma), where gamma is chosen to hit the sparsity.)

Phase 2 -- saliency-aware unlearning fine-tuning
    Continue training ONLY the masked weights on the randomly-labelled forget
    data with a very low learning rate (--lr). Non-salient weights stay frozen,
    which preserves the retained speakers' behaviour:
        θ_u = m * (θ_o + Δθ) + (1 - m) * θ_o
    Optionally, with --alpha > 0, a few retained-speaker batches (correct
    labels) are mixed in as a regularizer.

Checkpoints are written every --save-every epochs as G_unlearn_<epoch>.pth and
can be passed directly to `unlearning_evaluatioin.py --unlearned <ckpt>`.

Usage
-----
  python train_unlearn_salun.py -c configs/vctk_base.json \
      -m vctk_unlearn_salun \
      --pretrained pretrained/pretrained_vctk.pth \
      --forget-speaker p231 \
      --lr 1e-5 --epochs 10 --save-every 2 --sparsity 0.1
"""

import argparse
import collections
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioSpeakerCollate
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
from models import SynthesizerTrn
from losses import kl_loss
from text import cleaned_text_to_sequence
from text.symbols import symbols

VCTK_ROOT = os.path.join("DUMMY2", "wav48_silence_trimmed")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def _real_vctk_path(fl_path):
    """'DUMMY2/p231/p231_128.wav' -> '<VCTK_ROOT>/p231/p231_128_mic1.flac'"""
    spk = fl_path.split("/")[1]
    fid = os.path.basename(fl_path)[:-4]
    cand = os.path.join(VCTK_ROOT, spk, fid + "_mic1.flac")
    return cand if os.path.exists(cand) else fl_path


def _train_entries():
    """All (real_audio_path, sid, cleaned_text) from the train filelist."""
    entries = []
    fn = "filelists/vctk_audio_sid_text_train_filelist.txt.cleaned"
    with open(fn, encoding="utf-8") as f:
        for line in f:
            path, sid, text = line.rstrip("\n").split("|")
            real = _real_vctk_path(path)
            if not os.path.exists(real):
                continue
            entries.append((real, int(sid), text))
    return entries


def build_forget_entries(forget_sid, train_entries, random_labels=True, seed=0):
    """Forget data for SalUn: the forget speaker's audio, labelled with a
    random *other* speaker's sid (the 'random labeling' part of SalUn)."""
    other_sids = sorted({sid for _, sid, _ in train_entries} - {forget_sid})
    assert other_sids, "forget sid %d not present in the train set" % forget_sid
    rng = np.random.RandomState(seed)
    entries = []
    for path, sid, text in train_entries:
        if sid == forget_sid:
            label = int(rng.choice(other_sids)) if random_labels else sid
            entries.append((path, label, text))
    return entries


def build_retain_entries(forget_sid, train_entries, num_retain, per_spk=8, seed=0):
    """A small held-in retain set (correct labels) for the --alpha regularizer."""
    by_spk = collections.defaultdict(list)
    for path, sid, text in train_entries:
        by_spk[sid].append((path, sid, text))
    cand = sorted({sid for sid, lst in by_spk.items()
                   if sid != forget_sid and len(lst) >= per_spk})
    rng = np.random.RandomState(seed)
    rng.shuffle(cand)
    entries = []
    for sid in cand[:num_retain]:
        entries.extend(by_spk[sid][:per_spk])
    return entries


class ForgetSpeakerDataset(torch.utils.data.Dataset):
    """Minimal dataset for one speaker's training data.

    Reuses the repo's text / spectrogram pipeline but reads the real .flac
    audio directly (the stock loader cannot handle the .flac spec caching).
    `entries` are (audio_path, label_sid, cleaned_text) tuples.
    """
    def __init__(self, entries, hps):
        self.entries = entries
        self.hps = hps
        self.add_blank = hps.data.add_blank
        self.max_wav_value = hps.data.max_wav_value
        self.sampling_rate = hps.data.sampling_rate
        self.filter_length = hps.data.filter_length
        self.hop_length = hps.data.hop_length
        self.win_length = hps.data.win_length
        self._spec_cache = {}
        self._resampler = {}

    def get_audio(self, filename):
        if filename in self._spec_cache:
            return self._spec_cache[filename]
        spec_path = filename + ".spec.pt"
        # soundfile (not torchaudio: its torchcodec backend is broken on this box)
        wav, sr = sf.read(filename)
        wav = torch.FloatTensor(wav if wav.ndim == 1 else wav.mean(-1)).unsqueeze(0)
        if sr != self.sampling_rate:                    # VCTK is 48k; model wants 22050
            if (sr, self.sampling_rate) not in self._resampler:
                self._resampler[(sr, self.sampling_rate)] = \
                    torchaudio.transforms.Resample(sr, self.sampling_rate)
            wav = self._resampler[(sr, self.sampling_rate)](wav)
        audio_norm = wav / self.max_wav_value
        if os.path.exists(spec_path):
            spec = torch.load(spec_path)
        else:
            spec = torch.squeeze(
                spectrogram_torch(audio_norm, self.filter_length, self.sampling_rate,
                                  self.hop_length, self.win_length, center=False), 0)
            torch.save(spec, spec_path)
        self._spec_cache[filename] = (spec, audio_norm)
        return spec, audio_norm

    def get_text(self, text):
        text_norm = cleaned_text_to_sequence(text)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        return torch.LongTensor(text_norm)

    def __getitem__(self, idx):
        path, sid, text = self.entries[idx]
        spec, wav = self.get_audio(path)
        return self.get_text(text), spec, wav, torch.LongTensor([sid])

    def __len__(self):
        return len(self.entries)


# --------------------------------------------------------------------------
# model / loss
# --------------------------------------------------------------------------

def build_model(hps):
    return SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )


def generator_loss(model, batch, hps, device):
    """Generator-only reconstruction loss on one batch (no discriminator):
    mel L1 + KL(prior) + duration, same weights as train_ms.py."""
    x, x_lengths, spec, spec_lengths, y, y_lengths, sid = [t.to(device) for t in batch]
    y_hat, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = \
        model(x, x_lengths, spec, spec_lengths, sid)

    mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels,
                            hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
    y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
    y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), hps.data.filter_length,
                                      hps.data.n_mel_channels, hps.data.sampling_rate,
                                      hps.data.hop_length, hps.data.win_length,
                                      hps.data.mel_fmin, hps.data.mel_fmax)
    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask) * hps.train.c_kl
    loss_dur = torch.sum(l_length.float())
    total = loss_mel + loss_kl + loss_dur
    return total, {"mel": loss_mel.item(), "kl": loss_kl.item(), "dur": loss_dur.item()}


# --------------------------------------------------------------------------
# SalUn phase 1: weight saliency mask
# --------------------------------------------------------------------------

def compute_saliency_mask(model, loader, hps, device, sparsity):
    """SalUn weight saliency: mean |grad| of the forget loss per weight, then
    hard-threshold to keep the top `sparsity` fraction."""
    model.train()
    sal = {}
    n_batches = 0
    for batch in loader:
        model.zero_grad()
        total, _ = generator_loss(model, batch, hps, device)
        total.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                g = p.grad.detach().abs()
                sal[name] = sal.get(name) + g if name in sal else g
        n_batches += 1

    mask = {}
    n_kept = n_total = 0
    for name, g in sal.items():
        g = g / max(n_batches, 1)
        flat = g.reshape(-1)
        k = max(1, int(sparsity * flat.numel()))
        thr = flat.sort(descending=True).values[k - 1]
        m = (g >= thr).float()
        mask[name] = m
        n_kept += int(m.sum().item())
        n_total += int(flat.numel())
    print("Saliency mask: kept %d/%d weights (%.2f%%)"
          % (n_kept, n_total, 100.0 * n_kept / max(n_total, 1)))
    return mask


def zero_non_salient_grads(model, mask):
    """Only the salient weights get updated; everything else stays frozen."""
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if name in mask:
            p.grad *= mask[name].to(p.grad.device)
        else:
            p.grad.zero_()


# --------------------------------------------------------------------------
# SalUn phase 2: low-LR fine-tuning on the forget set
# --------------------------------------------------------------------------

def save_unlearn_checkpoint(model, optim, epoch, lr, path):
    torch.save({
        "model": model.state_dict(),
        "iteration": epoch,
        "learning_rate": lr,
        "optimizer": optim.state_dict(),
    }, path)
    print("  saved %s" % path)


def run_unlearning(hps, model, optim, forget_loader, retain_loader, mask, args, device):
    model.train()
    retain_iter = iter(retain_loader) if retain_loader is not None else None
    for epoch in range(1, args.epochs + 1):
        tot_loss, n = 0.0, 0
        parts_sum = collections.defaultdict(float)
        for batch in forget_loader:
            loss, parts = generator_loss(model, batch, hps, device)
            if retain_iter is not None and args.alpha > 0:
                try:
                    rb = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    rb = next(retain_iter)
                loss_r, parts_r = generator_loss(model, rb, hps, device)
                loss = loss + args.alpha * loss_r
                for k, v in parts_r.items():
                    parts_sum["r_" + k] += v
            optim.zero_grad()
            loss.backward()
            zero_non_salient_grads(model, mask)
            optim.step()
            tot_loss += loss.item()
            for k, v in parts.items():
                parts_sum[k] += v
            n += 1
        lr_now = optim.param_groups[0]["lr"]
        avg = tot_loss / max(n, 1)
        print("Epoch %3d/%d  loss_f=%.4f  %s  lr=%.2e"
              % (epoch, args.epochs, avg,
                 " ".join("%s=%.3f" % (k, v / n) for k, v in sorted(parts_sum.items())),
                 lr_now))

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_unlearn_checkpoint(
                model, optim, epoch, lr_now,
                os.path.join(args.model_dir, "G_unlearn_%d.pth" % epoch))
            save_unlearn_checkpoint(
                model, optim, epoch, lr_now,
                os.path.join(args.model_dir, "G_unlearn_last.pth"))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="SalUn speaker unlearning for VITS (VCTK)")
    ap.add_argument("-c", "--config", default="configs/vctk_base.json")
    ap.add_argument("-m", "--model-dir", default="vctk_unlearn_salun",
                    help="where checkpoints / masks are saved")
    ap.add_argument("--pretrained", default="pretrained/pretrained_vctk.pth")
    ap.add_argument("--forget-speaker", required=True,
                    help="target speaker to forget, e.g. 'p231' or a numeric sid")
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="very low learning rate for the unlearning fine-tune "
                         "(pretraining used 2e-4; 1e-5 is 20x lower)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=2,
                    help="save a checkpoint every N epochs")
    ap.add_argument("--sparsity", type=float, default=0.1,
                    help="SalUn: fraction of salient weights to update (top-rho%)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="weight of the retain-set regularizer (0 disables it)")
    ap.add_argument("--num-retain", type=int, default=8,
                    help="retain speakers sampled for the --alpha regularizer")
    ap.add_argument("--no-random-labels", action="store_true",
                    help="disable SalUn random labeling (use the true sid instead)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print("device:", device)

    hps = utils.get_hparams_from_file(args.config)
    train_entries = _train_entries()
    print("train entries:", len(train_entries))

    # resolve the forget speaker (pXXX or numeric sid)
    sids_in_train = sorted({sid for _, sid, _ in train_entries})
    if str(args.forget_speaker).startswith("p"):
        spk = str(args.forget_speaker)
        sid = None
        for path, s, _ in train_entries:
            if os.path.basename(os.path.dirname(path)) == spk:
                sid = s
                break
        assert sid is not None, "unknown speaker %s in train set" % spk
    else:
        sid = int(args.forget_speaker)
        assert sid in sids_in_train, "sid %d not in train set" % sid
    print("forget speaker: sid=%d" % sid)

    forget_entries = build_forget_entries(
        sid, train_entries, random_labels=not args.no_random_labels, seed=args.seed)
    print("forget entries (random-labelled):", len(forget_entries))
    forget_dataset = ForgetSpeakerDataset(forget_entries, hps)
    forget_loader = DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, collate_fn=TextAudioSpeakerCollate())

    retain_loader = None
    if args.alpha > 0:
        retain_entries = build_retain_entries(sid, train_entries, args.num_retain, seed=args.seed)
        retain_loader = DataLoader(ForgetSpeakerDataset(retain_entries, hps),
                                   batch_size=args.batch_size, shuffle=True,
                                   num_workers=0, collate_fn=TextAudioSpeakerCollate())
        print("retain entries (regularizer):", len(retain_entries))

    os.makedirs(args.model_dir, exist_ok=True)

    # load the pretrained model
    net_g = build_model(hps)
    net_g, _, _, _ = utils.load_checkpoint(args.pretrained, net_g, None)
    net_g = net_g.to(device)

    # SalUn phase 1: weight saliency mask
    print("Phase 1: computing weight saliency mask ...")
    mask = compute_saliency_mask(net_g, forget_loader, hps, device, args.sparsity)
    torch.save(mask, os.path.join(args.model_dir, "saliency_mask.pt"))
    print("mask saved -> %s/saliency_mask.pt" % args.model_dir)

    # SalUn phase 2: masked low-LR fine-tuning on the forget set
    optim = torch.optim.AdamW(net_g.parameters(), lr=args.lr,
                              betas=hps.train.betas, eps=hps.train.eps)
    print("Phase 2: unlearning fine-tune (lr=%.1e, epochs=%d, sparsity=%.2f, alpha=%.2f)"
          % (args.lr, args.epochs, args.sparsity, args.alpha))
    run_unlearning(hps, net_g, optim, forget_loader, retain_loader, mask, args, device)
    print("Done. Checkpoints in %s/  (feed G_unlearn_*.pth to "
          "unlearning_evaluatioin.py --unlearned <ckpt>)" % args.model_dir)


if __name__ == "__main__":
    main()
