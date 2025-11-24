import csv 
import numpy as np 
import pandas as pd 

df = pd.read_csv("data.csv") 
x1_class1 = [] 
x1_class2 = [] 
x2_class1 = [] 
x2_class2 = [] 
x3_class1 = [] 
x3_class2 = [] 
x4_class1 = [] 
x4_class2 = [] 

with open("data.csv", "r") as f: 
    reader = csv.DictReader(f) 
    for row in reader: 
        if row["C"] == "1": 
            x1_class1.append(float(row["X1"])) 
            x2_class1.append(float(row["X2"])) 
            x3_class1.append(float(row["X3"])) 
            x4_class1.append(float(row["X4"])) 
        else: 
            x1_class2.append(float(row["X1"])) 
            x2_class2.append(float(row["X2"])) 
            x3_class2.append(float(row["X3"])) 
            x4_class2.append(float(row["X4"])) 

# Хэвийн тархалттай X1 хувьсагчийн тархалтын 
# # μ дунджийн үнэлэлтийг дор бичиж оруулна. 

avg1 = sum(x1_class1) / len(x1_class1) 
avg2 = sum(x1_class2) / len(x1_class2) 
print("Class 1 x1 avg: ", avg1) 
print("Class 2 x1 avg: ", avg2) 
print("--------------------") 

# Хэвийн тархалттай X1 хувьсагчийн тархалтын 
# # σ стандарт хазайлтын үнэлэлтийг дор бичиж оруулна. 
# # Түүврийн стандарт хазайлт олоход Бесселийн засвар 
# # бүхий S=√1n−1∑i=1n(Xi−X¯)2 томьёо ашиглана. 
 
x1_class1_np = df[df["C"] == 1]["X1"].values 
x1_class2_np = df[df["C"] == 2]["X1"].values 
sigma1 = np.std(x1_class1_np, ddof=1) 
sigma2 = np.std(x1_class2_np, ddof=1) 
print("Class 1 σ =", sigma1) 
print("Class 2 σ =", sigma2) 
print("--------------------") 

# Бином тархалттай X2 хувьсагчийн 
# тархалтын p параметрийн үнэлэлтийг дор бичиж оруулна. 
# Энд X2 хувьсагчийн тархалтын n параметрийн утгыг 
# нэгтэй тэнцүү буюу уг хувьсагчийг Бернуллийн тархалттай гэж үзнэ.

p1 = np.sum(x2_class1) / len(x2_class1) 
p2 = np.sum(x2_class2) / len(x2_class2) 
print("Class 1 X2 p:", p1) 
print("Class 2 X2 p:", p2) 
print("--------------------") 

# Геометр тархалттай X3 хувьсагчийн тархалтын p параметрийн 
# үнэлэлтийг дор бичиж оруулна. 
# X3 хувьсагчийг ангиллаар салгаж авах 

x3_class1 = df[df["C"] == 1]["X3"].values 
x3_class2 = df[df["C"] == 2]["X3"].values 

# Моментийн аргаар p үнэлгээ 
p1 = 1 / (1 + np.mean(x3_class1)) 
p2 = 1 / (1 + np.mean(x3_class2)) 
print("Class 1 X3 p (moment estimate):", p1) 
print("Class 2 X3 p (moment estimate):", p2) 
print("--------------------") 

# Илтгэгч тархалттай X4 хувьсагчийн тархалтын 
# λ эрчмийн параметрийн үнэлэлтийг дор бичиж оруулна. 
# X4 хувьсагчийг ангиллаар салгах 

x4_class1 = df[df["C"] == 1]["X4"].values 
x4_class2 = df[df["C"] == 2]["X4"].values 

# λ үнэлгээ = 1 / жишээний дундаж 
lambda1 = 1 / np.mean(x4_class1) 
lambda2 = 1 / np.mean(x4_class2) 
print("Class 1 X4 λ:", lambda1) 
print("Class 2 X4 λ:", lambda2) 
print("--------------------") 

#test 
# Gaussian likelihood 

x = -2.11 
mean = np.mean(x1_class1)
print("sigma: ", sigma1)
print("mean: ", mean)
p_x1_c1 = (1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sigma1)**2)
print(f"P(X1|C1) = {p_x1_c1:.6f}")

x = 1.32 
mean = np.mean(x4_class1)
sigma4 = np.std(x4_class1, ddof=1)
print("sigma: ", sigma4)
avg4 = sum(x4_class1) / len(x4_class1) 
print("mean: ", avg4)
p_x1_c1 = (1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sigma1)**2)
print(f"P(X1|C1) = {p_x1_c1:.6f}")

