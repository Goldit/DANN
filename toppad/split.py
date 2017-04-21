from random import shuffle

num_test_file = 4000
synf = open('synthetic.txt').readlines()
train = open('train.txt', 'w')
test  = open('val.txt', 'w')
shuffle(synf)

for index, line in enumerate(synf):
    if index < num_test_file:
        test.write(line)
    else:
        train.write(line)

train.close()
test.close()
    

