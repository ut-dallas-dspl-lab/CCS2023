1. To attack linear models, run: 

        python linearProg.py

2. To attack non-linear models:

**Foolbox library change for our attacks**:\
    Non-linear foolbox library (all the changes): https://github.com/VibhaBelavadi/foolbox/tree/l2-mc \
    **Foolbox files modified in foolbox library**:\
        base.py: Base class for all attacks\
        projected_gradient_descent.py: Class for projected gradient descent\
        gradient_descent.py: super class for projected gradient descent\
        criteria.py: Misclassification criteria for multi-model attack

Example:

Steps to run attacks on CelebA:

1. Download the CelebA data, make sure the root is "data/celeba"

2. To train your own CelebA models, run: 

        python celeba_model.py --label LABEL --train yes 

   where LABEL can be any of {gender, attractive, young, glasses}.

3. You can also attack using readily trained models located in "models_celeba" and "ad_celeba_models". To attack resnet50, run:  

        python fb_pytorch_pgd_multiconcept.py 

 To attack adversarially trained Mobilnet, run: 

        python fb_pytorch_pgd_multiconcept.py --model ad_train 

   

4. To display the final results, for attacking resnet50 run:

        python display_results.py 

   for attacking adversarially trained Mobilenet run:  

        python display_results.py --traintype ad_train --modeltype mobile

