# Menstruation_pro

### step one: Menstruation_Classify_with_image

 1. run **show_result.py** to see our result.
 2. run **train_weight_QT.py** and **train_weight_ZQ.py** to train your model.
 3. run **test_combat_QT and ZQ_prob.py** to test your model.


### step two: Menstruation_Classify_with_image_and_EHR

1. you could run **show_result.ipynb** to see our result.
2. you could run **net_test.ipynb** to train and test our model.(with EHR and image)
   (if you want to trian your model, you should trian your model in **Menstruation_Classify_with_image** before)
    - use your model (step one) to get softmax value of images in  HER_训练集——FZ——其他切面.xlsx and HER_训练集——FZ——纵切.xlsx,then combine with their Menstrual cycle data.
    - use your model (step one) to get softmax value of images in  测试集——FZ——其他切面(一一对应).xlsx and 测试集——FZ——纵切(一一对应).xlsx,then combine with their Menstrual cycle data)
3. you could run **train_and_test.ipynb** to get a new model (only with EHR).
   
   
### Plot Result

1. run **plot_show.ipynb** to plot our ROC Curve.

### Data Download
1. https://drive.google.com/file/d/1dIgO0Ys4jfCLMYiuMnT6pQLDZwamyb5F/view?usp=sharing
2. github: https://github.com/CuiRobert/Menstruation_pro

