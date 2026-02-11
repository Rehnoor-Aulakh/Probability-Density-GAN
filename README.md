# GAN-Based PDF Estimation for Transformed Variable

## 1. Transformation Parameters
The assignment transformation is:

\[
z = x + a_r\sin(b_r x), \quad
 a_r = 0.5\,(r \bmod 7), \quad
 b_r = 0.3\,((r \bmod 5)+1)
\]

Run used:
- `r = 102317137`
- `a_r = 3.0`
- `b_r = 0.9`

## 2. Data and Assumption for `x`
`data.csv` has multiple numeric columns, not a single `x` column.
This solution treats one chosen numeric column as `x`.

Run used:
- `x = so2`
- valid samples: `401,096`
- train samples used: `50,000`

## 3. GAN Architecture
Implemented in `gan_pdf_estimation.py` using TensorFlow/Keras.

Generator:
- Input noise: `N(0,1)` with dimension `8`
- Dense(32) + LeakyReLU
- Dense(64) + LeakyReLU
- Dense(32) + LeakyReLU
- Dense(1) output

Discriminator:
- Input: scalar `z`
- Dense(64) + LeakyReLU + Dropout(0.2)
- Dense(32) + LeakyReLU + Dropout(0.2)
- Dense(1, sigmoid)

Training setup:
- Steps: `2000`
- Batch size: `256`
- Discriminator updates per generator update: `2`
- Loss: binary cross-entropy
- Label smoothing/noisy fake labels for stability

## 4. PDF Approximation from Generator Samples
After training, generated `100,000` synthetic samples `z_f`.
Estimated density using:
- Histogram (density normalized)
- KDE (`scipy.stats.gaussian_kde`)

Output files:
- PDF plot: `outputs_final/gan_pdf_plot.png`
- Training loss: `outputs_final/training_losses.png`
- Full summary: `outputs_final/summary.txt`
- Metrics JSON: `outputs_final/run_info.json`

## 5. Observations
From `outputs_final/run_info.json`:
- KS statistic: `0.0926`
- Wasserstein distance: `1.0598`
- JS divergence: `0.0111`
- Real mode count (KDE peaks): `5`
- Generated mode count (KDE peaks): `1`

Interpretation:
- Mode coverage: weak (mode collapse present)
- Training stability: stable loss curves in late training
- Quality of generated distribution: moderate global fit with missing minor modes

## 6. Re-run Command
Use this command to regenerate results:

```bash
python gan_pdf_estimation.py --roll-number 102317137 --x-column so2 --steps 2000 --d-steps 2 --batch-size 256 --train-samples 50000 --generate-samples 100000 --output-dir outputs_final
```

Then submit:
- `a_r, b_r` from `outputs_final/summary.txt`
- GAN architecture from `outputs_final/summary.txt`
- PDF plot `outputs_final/gan_pdf_plot.png`
- observations from `outputs_final/summary.txt`
