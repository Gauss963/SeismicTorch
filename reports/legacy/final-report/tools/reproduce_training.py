"""Recover epoch telemetry without changing the released checkpoint or using test clips."""
from argparse import Namespace
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset
from seismictorch.qsis.candidate import recut_training_clip
from seismictorch.training import pilot

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / 'report-data'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    receipt_path = ROOT / 'models/qsis/current/training.json'
    receipt = json.loads(receipt_path.read_text())
    init = ROOT / 'runs/qsis-benchmark/cwa-pretrain-42/best.pth'
    checkpoint = ROOT / 'models/qsis/current/model.pth'
    assert digest(init) == receipt['init_sha256']
    assert digest(checkpoint) == receipt['checkpoint_sha256']
    assert torch.backends.mps.is_available(), 'Exact reproduction requires the original MPS device'
    rows = receipt['sources']
    clips = [recut_training_clip(row, receipt['waveform_filter'], receipt['orientation']) for row in rows]
    inputs = torch.from_numpy(np.stack(clips))
    targets = torch.tensor([r['label'] for r in rows])
    data = TensorDataset(inputs, targets)
    history = []
    original = pilot.run_epoch
    device = torch.device('mps')

    def observe(model, loader, device, criterion, optimizer=None, train_mode=True):
        result = original(model, loader, device, criterion, optimizer, train_mode)
        state = torch.get_rng_state()
        mps_state = torch.mps.get_rng_state()
        was_training = model.training
        model.eval()
        with torch.inference_mode():
            logits = torch.cat([model(batch.to(device)).cpu() for batch in inputs.split(64)])
            loss = torch.nn.functional.cross_entropy(logits, targets).item()
            predicted = logits.argmax(1)
            matrix = torch.bincount(3 * targets + predicted, minlength=9).reshape(3, 3)
            recall = matrix.diag() / matrix.sum(1)
        model.train(was_training)
        torch.set_rng_state(state)
        torch.mps.set_rng_state(mps_state)
        row = {'epoch': len(history) + 1, 'lr': optimizer.param_groups[0]['lr'],
               'train_augmented': result,
               'train_clean': {'loss': loss, 'accuracy': (predicted == targets).float().mean().item(),
                               'balanced_accuracy': recall.mean().item(), 'confusion_matrix': matrix.tolist()}}
        history.append(row)
        print(f"Epoch {row['epoch']:2}: augmented loss {result['loss']:.4f}; clean training accuracy {row['train_clean']['accuracy']:.4f}", flush=True)
        return result

    start = time.monotonic()
    pilot.run_epoch = observe
    try:
        model = pilot.fit(data, list(range(len(rows))), torch.load(init, weights_only=True, map_location='cpu'),
                          device, Namespace(model=receipt['model'], seed=receipt['seed'], epochs=receipt['epochs']),
                          balance=True, augment=True)
    finally:
        pilot.run_epoch = original
    expected = torch.load(checkpoint, weights_only=True, map_location='cpu')
    actual = {k: v.cpu() for k, v in model.state_dict().items()}
    mismatches = [key for key in expected if not torch.equal(expected[key], actual[key])]
    if mismatches:
        raise RuntimeError(f'Reproduction differs from released weights: {mismatches}; do not publish these curves as exact')
    OUT.mkdir(exist_ok=True)
    payload = {'purpose': 'Instrumented exact replay; training-only telemetry, no validation/test monitoring',
               'checkpoint_sha256': digest(checkpoint), 'training_receipt_sha256': digest(receipt_path),
               'init_sha256': digest(init), 'training_input_sha256': hashlib.sha256(inputs.numpy().tobytes()).hexdigest(),
               'device': 'mps', 'torch_version': torch.__version__, 'seed': receipt['seed'],
               'source_count': len(rows), 'all_state_tensors_equal': True, 'state_tensor_count': len(expected),
               'elapsed_seconds': time.monotonic() - start, 'history': history,
               'loss_note': 'Augmented loss is the existing run_epoch batch-weighted aggregate of weighted CE; clean loss is ordinary unweighted CE on unaugmented fitting clips.'}
    (OUT / 'final-training-replay.json').write_text(json.dumps(payload, indent=2) + '\n')
    print('Verified all released state tensors exactly; wrote final-training-replay.json')

if __name__ == '__main__':
    main()
