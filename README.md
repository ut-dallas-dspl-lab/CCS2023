The repository contains a foolbox submodule (in "lib") with custom attacks presented in our CCS paper entitled "Attack Some while Protecting Others: Selective Attack Strategies for Attacking and Protecting Multiple Concepts".
https://dl.acm.org/doi/10.1145/3576915.3623177

To recursively clone the repository, run:

        git clone --recursive https://github.com/VentiAwake/CCS2023.git 

You can create a Conda environment with the supplied yaml file "atk.yaml":
        
        conda env create --name NAME --file atk.yaml 

where NAME is the name of the environment. Activate the environment, everything should run following the instructions below.




1. To attack linear models, run: 

        python linearProg.py

2. To attack non-linear models:

Steps to run attacks on CelebA:

1.) Download the CelebA data to the non-linear/CelebA folder, make sure the root is "data/celeba"

2.) To train your own CelebA models, run: 

        python celeba_model.py --label LABEL --train yes 

   where LABEL can be any of {gender, attractive, young, glasses}.

3.) You can also attack using readily trained models located in "models_celeba" and "ad_celeba_models". To attack resnet50, run:  

        python fb_pytorch_pgd_multiconcept.py 

   To attack adversarially trained Mobilnet, run: 

        python fb_pytorch_pgd_multiconcept.py --model ad_train 

   
4.) To display the final results, for attacking resnet50 run:

        python display_results.py 

   for attacking adversarially trained Mobilenet run:  

        python display_results.py --traintype ad_train --modeltype mobile


Steps to run attacks on UTKFace:

1.) Download the UTKFace data to the non-linear/UTKFace folder, make sure the root is "utkcropped"


2.) To train your own UTKFace models, run: 

        jupyter execute UTKFace_train-age.ipynb UTKFace_train-gen.ipynb UTKFace_train-eth.ipynb UTKFace_train_utils.py

  * Trained models are available at: https://utdallas.box.com/s/903f929uqv4n8dqj1qfd9uqm45sgs255

3.) To attack, run:

        python fb_pytorch_pgd_utk.py  

4.) To display the final results, run:

        python display_results_utkface.py 

