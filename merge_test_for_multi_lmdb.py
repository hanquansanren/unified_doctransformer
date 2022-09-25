# coding=utf-8

import lmdb

# 将两个lmdb文件合并成一个新的lmdb
def merge_lmdb(lmdb_list, result_lmdb):
    print ('Merge start!')
    env_list = []
    txn_list = []
    db_list = []
    env = lmdb.open(result_lmdb, map_size=1099511627776)
    for i in range(len(lmdb_list)):
        env_list.append(lmdb.open(lmdb_list[i]))

    for j in range(len(env_list)):
        txn_list.append(env_list[j].begin(write=True))

    # for jj in range(len(txn_list)):
    #     txn_list[jj].delete(b'__keys__')
    #     txn_list[jj].delete(b'__len__')
    #     txn_list[jj].commit() 
    #     env_list[jj].begin()

    for j in range(len(env_list)):
        print(env_list[j].stat())
    
    for k in range(len(txn_list)):
        db_list.append(txn_list[k].cursor())


    txn_output = env.begin(write=True)
    count = 0
    idx = 0
    idx_flag=0
    for m in range(len(db_list)):
        for (key, value) in db_list[m]:
            txn_output.put((lmdb_list[m][-6]+'_').encode()+(key.decode()[0:7] + str(idx) + '_' +key.decode().split("_")[3]).encode(), value)
            count = count + 1
            idx_flag = idx_flag + 1
            if idx_flag%4==0:
                idx = idx + 1
            if(count % 1000 == 0):
                txn_output.commit()
                count = 0
                txn_output = env.begin(write=True)

        if(count % 1000 != 0):
            txn_output.commit()
            count = 0
            txn_output = env.begin(write=True)
        print("finish {}".format(lmdb_list[m]))
        print (env.stat())



    print (env.stat())
    env.close()
    for j in range(len(env_list)):
        env_list[j].close()
    print ('Merge success!')


def main():
    lmdb_filename_list=[]
    with open("./dataset/lmdb.txt","r") as f:
        for ii in range(10):
            lmdb_filename_list.append(f.readline().strip())
    output_lmdb_root =  './dataset/biglmdb/merged_0.lmdb'
    merge_lmdb(lmdb_filename_list, output_lmdb_root)


















if __name__ == '__main__':
    main()




