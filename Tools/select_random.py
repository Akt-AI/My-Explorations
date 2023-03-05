import random


random.seed(20)
images = []

foo = ['a', 'b', 'c', 'd', 'e']

for i in range(1000):
    #print(random.choice(foo))
    random_img = random.choice(foo)
    images.append(random_img)
    
print(images)

