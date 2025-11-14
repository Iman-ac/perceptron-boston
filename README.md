Perceptron Regressor on Boston Housing Dataset

توضیح:
دیتاست شامل 506 نمونه و 13 ویژگی
ویژگی های انتخابی: متوسط اتاق ها (مساحت(RM)) و سن خانه(AGE) 
مدل:simple Perceptron with  Gradient Descent (MSE loss).
نتایج:MSE Train: 40.33، MSE Test: 38.47. و وزن‌ها: [0.58و -0.21].
تحلیل همبستگی RM: 0.7 (مثبت)، AGE: -0.38 (منفی). مدل خطی خوب تعمیم می‌دهد.

 اجرا
python main.py

 وابستگی‌ها
pip install -r requirements.txt