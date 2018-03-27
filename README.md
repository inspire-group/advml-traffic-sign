# DARTS: Deceiving Autonomous Cars with Toxic Signs

Website: http://inspiregroup.deptcpanel.princeton.edu/darts/

The code in this repository is associated with the paper _[DARTS: Deceiving Autonomous Cars with Toxic Signs](https://arxiv.org/abs/1802.06430)_  and its earlier extended abstract _[Rogue Signs: Deceiving Traffic Sign Recognition with Malicious Ads and Logos](https://arxiv.org/pdf/1801.02780.pdf)_ , a research project under the INSPIRE group in the Electrical Engineering Department at Princeton University. It is the same code that we used to run the experiments, but excludes some of the run scripts as well as the datasets used. Please download the dataset in pickle format [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip), or visit the original [website](http://benchmark.ini.rub.de/?section=home&subsection=news) for GTSRB and GTSDB datasets.  

## Files Organization
The main implementation is in [./lib](./lib) containing:
- [utils.py](./lib/utils.py): utility functions
- [attacks.py](./lib/attacks.py): previously proposed adversarial examples generation methods
- [keras_utils.py](./lib/keras_utils.py): define models in [Keras](https://keras.io/)
- [OptProjTran.py](./lib/OptProjTran.py): our optimization code for generating physicall robust adversarial examples
- [OptCarlini.py](./lib/OptCarlini.py): implementation of [Carlini-Wagner's attack](https://arxiv.org/abs/1608.04644)
- [RandomTransform.py](./lib/RandomTransform.py): implementation of random perspective transformation

For specific data/setup we used in our experiments:
- [images](./images): contains original images to generate the attacks 
  - [Original_Traffic_Sign_samples](./images/Original_Traffic_Sign_samples): original traffic signs for Adversarial Traffic Signs
  - [Logo_samples](./images/Logo_samples): original logos for Logo Attack
  - [Custom_Sign_samples](./images/Custom_Sign_samples): blank signs to be used as background for Custom Sign Attack
- [adv_signs](./adv_signs): contains some of the adversarial signs we produced, saved in pickle. Organized by types of the attacks: [Adversarial_Traffic_Signs](./adv_signs/Adversarial_Traffic_Signs), [Logo_Attacks](./adv_signs/Logo_Attacks), [Custom_Sign_Attacks](./adv_signs/Custom_Sign_Attacks), and [Lenticular](./adv_signs/Lenticular). Code to read the data is in [Run_Robust_Attack.ipynb](./Run_Robust_Attack.ipynb).
- [keras_weights](./keras_weights): contains the weight of multiple Keras models we used in the experiment
  - `weights_mltscl_dataaug.hdf5`: multi-scale CNN with data augmentation ("CNN A" in the paper)
  - `weights_cnn_dataaug.hdf5`: normal CNN with data augmentation ("CNN B" in the paper)
- For videos of our drive-by test, please visit the website listed above.

The main code we used to run the experiments is in [Run_Robust_Attack.ipynb](./Run_Robust_Attack.ipynb). It demonstrates our procedures and usage of the functions in the library. It also includes code that we used to run most of the experiments from generating the attacks to evaluating them in both virtual and physical settings.   
Examples of previously proposed adversarial examples generation methods are listed in [GTSRB.ipynb](./GTSRB.ipynb).  
Relevant parameters are set in a separate configure file called [parameters.py](./parameters.py).

## Contact
Comments and suggestions can be sent to Chawin Sitawarin (<chawins@princeton.edu>) and Arjun Nitin Bhagoji (<abhagoji@princeton.edu>).
