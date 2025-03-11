# PULSE: Proximal Unrolling for Learning Solutions in ECGI
```PULSE: Proximal Unrolling for Learning Solutions in ECGI```
This repository includes the code for inverse ECG reconstruction via Half Quadratic Splitting.

<div align="center">
    <img src="torso_voltage.gif" width="250px">
    <p><strong>Torso Voltage</strong></p>
</div>

<div align="center">
    <table>
        <tr>
            <td align="center"><strong>Reconstruction</strong></td>
            <td align="center"><img src="Tik.gif" width="150px"></td>
            <td align="center"><img src="MAP.gif" width="150px"></td>
            <td align="center"><img src="NN.gif" width="150px"></td>
            <td align="center"><img src="GT.gif" width="150px"></td>
        </tr>
        <tr>
            <td align="center"><strong>Method</strong></td>
            <td align="center"><strong>Tik</strong></td>
            <td align="center"><strong>MAP</strong></td>
            <td align="center"><strong>NN</strong></td>
            <td align="center"><strong>GT</strong></td>
        </tr>
        <tr>
            <td align="center">Regularization</td>
            <td align="center">Only spatial regularization</td>
            <td align="center">Data-driven spatial regularization</td>
            <td align="center">Learned spatiotemporal regularization</td>
            <td align="center">-</td>
        </tr>
    </table>
</div>



```bibtex
@article{ugurlu2024,
  author    = {Kutay Ugurlu and Gozde B. Akar and Yesim Serinagaoglu Dogrusoz},
  title     = {PULSE: Proximal Unrolling for Learning Solutions in ECGI},
  journal   = {IEEE Transactions on Biomedical Engineering},
  year      = {2024},
  note      = {Submitted for publication}
}
