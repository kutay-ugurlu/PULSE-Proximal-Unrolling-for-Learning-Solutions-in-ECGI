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
            <td><strong>Reconstruction</strong></td>
            <td><img src="Tik.gif" width="150px"></td>
            <td><img src="MAP.gif" width="150px"></td>
            <td><img src="NN.gif" width="150px"></td>
            <td><img src="GT.gif" width="150px"></td>
        </tr>
        <tr>
            <td><strong>Method</strong></td>
            <td><strong>Tik</strong></td>
            <td><strong>MAP</strong></td>
            <td><strong>NN</strong></td>
            <td><strong>GT</strong></td>
        </tr>
        <tr>
            <td> Regularization </td>
            <td> Only spatial regularization </td>
            <td> Data-driven spatial regularization </td>
            <td>Learned spatiotemporal regularization </td>
            <td> - </td>
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
