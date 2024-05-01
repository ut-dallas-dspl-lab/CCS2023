The repository contains a foolbox submodule (in "lib") with custom attacks presented in our CCS paper entitled "Attack Some while Protecting Others: Selective Attack Strategies for Attacking and Protecting Multiple Concepts". To recursively clone the repository, run:

        git clone --recursive https://github.com/VentiAwake/CCS2023.git 

1. To attack linear models, run: 

        python linearProg.py

2. To attack non-linear models:

Steps to run attacks on CelebA:

1.) Download the CelebA data, make sure the root is "data/celeba"

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

1.) Download the UTKFace data, make sure the root is "data/utkface"


2.) To train your own UTKFace models, run: 

        jupyter execute UTKFace_train-age.ipynb UTKFace_train-gen.ipynb UTKFace_train-eth.ipynb UTKFace_train_utils.py

    Trained models are available at: https://utdallas.box.com/s/903f929uqv4n8dqj1qfd9uqm45sgs255

3.) To attack, run:

        python fb_pgd_UTKFace.py 

4.) To display the final results, run:

        jupyter execute pgd_metrics_UTKFace-exp.ipynb

