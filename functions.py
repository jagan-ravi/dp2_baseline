from asyncio.log import logger
import random
import csv
import statistics
import numpy



#------------------------------------------#
#-----Random Worker Selection Strategy-----#
#------------------------------------------#


def round_worker(round,client,round_client):
    key=int((round*round_client)/client*round_client)*7
    #random.seed(key)
    round_key=random.sample(list(range(1000)),round)
    round_workers=[]
    for i in round_key:
        #random.seed(i)
        temp=random.sample(list(range(client)),round_client)
        temp.sort
        round_workers.append(temp)
    return round_workers
def random_worker_selection(num_of_worker_per_round,num_of_piosioned_worker_per_round,num_poisoned_workers,num_workers,epoch):
    random_workers_poisionous=[]
    random_workers_non_poisionous_with_replacement=[]
    random_workers_non_poisionous_without_replacement=[]
    for i in range(epoch):
        li1=random.sample(list(range(num_poisoned_workers)),num_of_piosioned_worker_per_round)
        li2=random.sample(list(range(num_poisoned_workers,num_workers)),num_of_worker_per_round)
        li3=random.sample(li2,num_of_worker_per_round-num_of_piosioned_worker_per_round)
        random_workers_poisionous.append(li1+li3)
        random_workers_non_poisionous_with_replacement.append(li2)
        random_workers_non_poisionous_without_replacement.append(li3)
    return random_workers_poisionous,random_workers_non_poisionous_with_replacement,random_workers_non_poisionous_without_replacement
    ##now RANDOM_WORKERS is a 3d list where:
    #index 0 hold random worker with poisoned clients
    #index 1 hold random worker with out poision and with replacement of the poisoned clients
    #index 2 hold random worker with out poision and without replacement of the poisoned clients



###poisoned worker selection
def poisioned_worker_selection(ROUND,KWARGS):
    return KWARGS["RANDOM_WORKERS"][ROUND][:KWARGS["NUM_POISONED_WORKERS"]]



#-----------------------------------------
#-----CSV genaration functions------------
#-----------------------------------------



def csv_gen(KWARGS,data):
    KWARGS_Copy=KWARGS.copy()
    del KWARGS_Copy["RANDOM_WORKERS"]
    with open('sheets/'+str(KWARGS["EXPID"])+'.csv', 'w') as file:
        writer = csv.writer(file)
        for row in KWARGS_Copy:
            writer.writerow([str(row),KWARGS_Copy[row]])
        writer.writerow([""])
        writer.writerow([""])
        writer.writerow([""])
        avgTestAcc=[]
        avgTrainAcc=[]
        avgSrcRcall=[]
        avgTargetRcall=[]
        avgSrcIndRcall=[]
        avgTargetIndRcall=[]
        avgRcallmal=[]
        avgRcallpure=[]
        writer.writerow(["Round","Random Workers","Test Avg Accuracy","Train Avg Accuracy","Source Avg recall","Targer Avg recall","Source individual recall","Target individual recall","Avg rcall including mal","Avg rcall Excluding mal","Client Train Acc Before Update","Client Train Acc After Update","Client Test acc After Update"])
        for ra in range(KWARGS["ROUNDS"]):
            round_data=data[ra]
            workers=KWARGS["RANDOM_WORKERS"][ra]
            avgTestAcc.append(round(statistics.fmean(round_data[2]),3))
            avgTrainAcc.append(round(statistics.fmean(round_data[1]),3))
            avgSrcIndRcall.append(axis_avg(round_data[3],KWARGS["LABELS_TO_REPLACE"]))
            avgTargetIndRcall.append(axis_avg(round_data[3],KWARGS["LABELS_TO_REPLACE_WITH"]))
            avgSrcRcall.append(round(statistics.fmean(avgSrcIndRcall[ra]),3))
            avgTargetRcall.append(statistics.fmean(avgTargetIndRcall[ra]))
            avgRcallmal.append(round(recall_avg(round_data[3],[]),3))
            avgRcallpure.append(round(recall_avg(round_data[3],KWARGS["LABELS_TO_REPLACE"]+KWARGS["LABELS_TO_REPLACE_WITH"]),3))
            writer.writerow([ra,workers,avgTestAcc[ra],avgTrainAcc[ra],avgSrcRcall[ra],avgTargetRcall[ra],avgSrcIndRcall[ra],avgTargetIndRcall[ra],avgRcallmal[ra],avgRcallpure[ra],round_data[0],acc_select_rand_client(round_data[1],workers),acc_select_rand_client(round_data[2],workers)])
        writer.writerow([["Average Test Accuracy",round(statistics.fmean(avgTestAcc),3)]])
        writer.writerow([["Average Train Accuracy",round(statistics.mean(avgTrainAcc),3)]])
        writer.writerow([["Average Src Rcall",round(statistics.fmean(avgSrcRcall),3)]])
        writer.writerow([["Average Tget Rcall",round(statistics.fmean(avgTargetRcall),3)]])
        writer.writerow([["Average Rcall with MAl",round(statistics.fmean(avgRcallmal),3)]])
        writer.writerow([["Average Rcall without mal",round(statistics.fmean(avgRcallpure),3)]])

def axis_avg(arr,col):
    ar=numpy.array(arr)
    avg=[]
    if col==[]:
        return [0]
    for i in col:
        avg.append(round(statistics.fmean(ar[:,i])*100,3))
    return avg
def recall_avg(arr,mal):
    ar=numpy.array(arr)
    for i in mal:
        ar[:,i]=0
    row_avg=[]
    lenmal=len(mal)
    for row in ar:
        row_avg.append((sum(row)/(len(row)-lenmal))*100)
    return statistics.fmean(row_avg)
def acc_select_rand_client(acc,clients):
    sel_acc=[]
    for i in clients:
        sel_acc.append(acc[i])
    return sel_acc
    