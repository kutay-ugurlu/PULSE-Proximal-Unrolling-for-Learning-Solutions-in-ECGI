# PULSE: Proximal Unrolling for Learning Solutions in ECGI
PULSE: Proximal Unrolling for Learning Solutions in ECGI  
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

# Citation

```bibtex
@article{ugurlu2024pulse,
  title={PULSE: A DL-assisted physics-based approach to the inverse problem of electrocardiography},
  author={Ugurlu, Kutay and Akar, Gozde B and Dogrusoz, Yesim Serinagaoglu},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2024},
  publisher={IEEE}
}
```
# How to run 
* Install dependencies with ``conda create --name pulse --file requirements.txt``
* Place the training data in [TrainingData](TrainingData) folder following the structure of ``.mat`` files. 
* Place the test data in [TestData](TrainingData) folder following the structure of ``.mat`` files. 
* Use [create_training_data.ipynb](create_training_data.ipynb) notebook to generate framework-compatible training and test data.
*  [generate_test_results_different_geoms.ipynb](generate_test_results_different_geoms.ipynb) is for geometric variation. 
* Once the data is ready, make a model selection from [models.py](models.py) and put your selection in [train_reg.py](train.py).
* Run ```python train_reg.py -bs <BATCHSIZE>``` command to start training, and the script will generate the results in [TestResults](TestResults) folder.

# Information on Data
The shared data is only a part of the actual data used in the study. To gain access to whole database, email [Yesim Serinagaoglu Dogrusoz](mailto:yserin@metu.edu.tr). For remaining questions, email [Kutay Ugurlu](mailto:kutay.ugurlu.1@gmail.com).


