"""
train_size = int(0.8 * len(full_dataset))
validation_size = len(full_dataset) - train_size
train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])

full_loader = DataLoader(full_dataset, batch_size=4,sampler = sampler_(full_dataset), pin_memory=True) 
train_loader = DataLoader(train_dataset, batch_size=4, sampler = sampler_(train_dataset))
val_loader = DataLoader(validation_dataset, batch_size=1, sampler = sampler_(validation_dataset))"""


# define a cross validation function
def crossvalid(model=None,criterion=None,optimizer=None,dataset=None,k_fold=5):
    
    train_score = pd.Series()
    val_score = pd.Series()
    
    total_size = len(dataset)
    fraction = 1/k_fold
    seg = int(total_size * fraction)
    # tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset 
    # index: [trll,trlr],[vall,valr],[trrl,trrr]
    for i in range(k_fold):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size
        #msg
        print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" 
               % (trll,trlr,trrl,trrr,vall,valr))
        
        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall,valr))
        
        train_set = torch.utils.data.dataset.Subset(dataset,train_indices)
        val_set = torch.utils.data.dataset.Subset(dataset,val_indices)
        
        #print(len(train_set),len(val_set))
        #print()
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=50,
                                          shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=50,
                                          shuffle=True, num_workers=4)
        train_acc = train(res_model,criterion,optimizer,train_loader,epoch=1)
        train_score.at[i] = train_acc
        val_acc = valid(res_model,criterion,optimizer,val_loader)
        val_score.at[i] = val_acc
    
    return train_score,val_score
        

train_score,val_score = crossvalid(res_model,criterion,optimizer,dataset=tiny_dataset)
results = []
for fold in range(total_fold):
    train_set, test_set = split_dataset(your_dataset, fold)
    model = MyModel()
    results.append(your_training_function(model, train_set, test_set)
print(mean(results))
