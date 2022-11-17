
# train="experiment/full_data/IPTV_0.7/train.txt"
# test="experiment/full_data/IPTV_0.7/test.txt"
# user_list=[]
# item_list=[]
# with open(train,"r") as f:
#     f.readline()
#     for cnt, l in enumerate(f):
#         ls = l.strip().split(" ")
#         user=ls[0]
#         item=ls[1]
#         timestamp=ls[2]
#         if int(user) not in user_list:user_list.append(int(user))
#         if int(item) not in item_list:item_list.append(int(item))
# user_list.sort(reverse=True)
# item_list.sort(reverse=True)
# f.close()



# path="ml-1m/ratings.dat"
# new_path="ml-1m/ml-1m.txt"
# new=open(new_path,"a")
# with open(path,"r") as f:
#     f.readline()
#     for cnt, l in enumerate(f):
#         # FORMAT: user, item, timestamp, state label, feature list
#         ls = l.strip().split("::")
#         user=ls[0]
#         item=ls[1]
#         timestamp=ls[3]
#         str=user+" "+item+" "+timestamp+" "+'0'+" "+'0'+" "+'0'
#         new.write('\n')
#         new.write(str)
#     f.close()


# path="ml-1m/ml-100k.csv"
# user_list=[]
# item_list=[]
# with open(path,"r") as f:
#     f.readline()
#     for cnt, l in enumerate(f):
#         # FORMAT: user, item, timestamp, state label, feature list
#         ls = l.strip().split("\t")
#         user=ls[0]
#         item=ls[1]
#         timestamp=ls[3]
#         if user not in user_list:user_list.append(user)
#         if item not in item_list: item_list.append(item)
#     f.close()



# path="ml-1m/ml-1m.txt"
# new_path="ml-1m/test.txt"
# new=open(new_path,"a")
# with open(path,"r") as f:
#     f.readline()
#     for cnt, l in enumerate(f):
#         # FORMAT: user, item, timestamp, state label, feature list
#         ls = l.strip().split(" ")
#         user=ls[0]
#         item=ls[1]
#         timestamp=ls[2]
#         if int(timestamp)>=975500000:
#             str=user+" "+item+" "+timestamp+" "+'0'+" "+'0'+" "+'0'
#             new.write('\n')
#             new.write(str)
#     f.close()


# path=open("ml-1m/ml-1m.txt")
# new_path="ml-1m/sorted-ml.txt"
# result=[]
# iter_f=iter(path)
# for line in iter_f:
#     if line[0]=='u':continue
#     result.append(line)
# path.close()
# result.sort(key=lambda x:float(x.split(' ')[2]),reverse=False)
# f=open(new_path,'w')
# f.writelines(result)
# f.close()

ratio=0.7
# num=2392010*ratio
# num=14816*ratio
num=1000208*ratio
train_file='train'+str(ratio)+'.txt'
test_file='test'+str(ratio)+'.txt'
count=0
# train="experiment/full_data/reddit_1000_random_0.7/trainAll.txt"
# new_path="experiment/full_data/reddit_1000_random_0.7/"+train_file
# test_path="experiment/full_data/reddit_1000_random_0.7/"+test_file
# train="experiment/full_data/IPTV_0.7/back/trainAll.txt"
# new_path="experiment/full_data/IPTV_0.7/back/"+train_file
# test_path="experiment/full_data/IPTV_0.7/back/"+test_file
train="ml-1m/sorted-ml.txt"
new_path="ml-1m/"+train_file
test_path="ml-1m/"+test_file
new=open(new_path,"a")
test=open(test_path,"a")
with open(train,"r") as f:
    row=f.readlines()
    for row in row:
        if count < num:
            new.write(row)
        if count>=num:
            test.write(row)
        count=count+1

# ratio=0.9
# # num=2392010*ratio*(1-0.333)
# num=14816*ratio*(1-0.222222222)
# train_file='train'+str(ratio)+'.txt'
# test_file='test'+str(ratio)+'.txt'
# count=0
# train="experiment/full_data/reddit_1000_random_0.7/data0.9.txt"
# new_path="experiment/full_data/reddit_1000_random_0.7/"+train_file
# test_path="experiment/full_data/reddit_1000_random_0.7/"+test_file
# # train="experiment/full_data/IPTV_0.7/back/data0.4.txt"
# # new_path="experiment/full_data/IPTV_0.7/back/"+train_file
# # test_path="experiment/full_data/IPTV_0.7/back/"+test_file
# new=open(new_path,"a")
# test=open(test_path,"a")
# with open(train,"r") as f:
#     row=f.readlines()
#     for row in row:
#         if count < num:
#             new.write(row)
#         if count>=num:
#             test.write(row)
#         count=count+1