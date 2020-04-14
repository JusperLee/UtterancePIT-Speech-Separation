## UtterancePIT-Speech-Separation

### According to funcwj's uPIT, the training code supporting multi-gpu is written, and the Dataloader is reconstructed.

### If you want to see the funcwj code, this is his repository link.     
[uPIT-for-speech-separation](https://github.com/funcwj/uPIT-for-speech-separation)

Demo Pages: [Results of pure speech separation model](https://www.likai.show/Pure-Audio/index.html)

### Accomplished goal
- [x] **Support Multi-GPU Training**
- [x] **Use the Dataloader Method That Comes With Pytorch**
- [x] **Provide Pre-Training Models**

### Python Library Version
- Pytorch==1.3.0
- tqdm==4.32.1
- librosa==0.7.1
- scipy==1.3.0
- numpy==1.16.4
- PyYAML==5.1.1

### How to Using This Repository
 
1. Generate dataset using [create-speaker-mixtures.zip](http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip) with WSJ0 or TIMI
2. Prepare scp file(The content of the scp file is "filename path")
    ```shell
     python create_scp.py
    ```
3. Prepare cmvn(Cepstral mean and variance normalization (CMVN) is a computationally efficient normalization technique for robust speech recognition.).
    ```shell
     #Calculated by the compute_cmvn.py script: 
     python compute_cmvn.py ./tt_mix.scp ./cmvn.dict
    ```
4. Modify the contents of yaml, mainly to modify the scp address, cmvn address. At the same time, the number of num_spk in run_pit.py is modified.
5. Training:
    ```shell
    sh train.sh
    ```

6. Inference:
    ```
    sh test.sh
    ```


### Reference

* Kolbæk M, Yu D, Tan Z H, et al. Multitalker speech separation with utterance-level permutation invariant training of deep recurrent neural networks[J]. IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP), 2017, 25(10): 1901-1913.
* https://github.com/funcwj/uPIT-for-speech-separation
