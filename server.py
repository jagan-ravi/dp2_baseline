from ast import arg
from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.utils import average_nn_parameters
from federated_learning.utils import convert_distributed_data_into_numpy
from federated_learning.utils import poison_data
from federated_learning.utils import identify_random_elements
from federated_learning.utils import save_results
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
from client import Client
from functions import poisioned_worker_selection
from functions import csv_gen

def train_subset_of_clients(epoch, args, clients,KWARGS,distributed_train_dataset):
    """
    Train a subset of clients per round.

    :param epoch: epoch
    :type epoch: int
    :param args: arguments
    :type args: Arguments
    :param clients: clients
    :type clients: list(Client)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    """
    # commented
    # kwargs = args.get_round_worker_selection_strategy_kwargs()
    # kwargs["current_epoch_number"] = epoch

    # random_workers = args.get_round_worker_selection_strategy().select_round_workers(
    #     list(range(args.get_num_workers())),
    #     poisoned_workers,
    #     kwargs)


    ##getting random worker
    random_workers=KWARGS["RANDOM_WORKERS"][epoch-1]
    ##Choosing poisoned worker
    poisoned_workers=poisioned_worker_selection(epoch-1,KWARGS)
    ##Genarating dataset
    distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)
    distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers,KWARGS)
    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())

    ##Training the Random Clients
    trainAccuracyOfRand=[0]
    # for client_idx in random_workers:
    #     args.get_logger().info("Training epoch #{} on client #{}", str(epoch), str(clients[client_idx].get_client_index()))
    #     acc,_=clients[client_idx].train(epoch,train_data_loaders[client_idx])
    #     trainAccuracyOfRand.append(acc)


    ##Averaging parameters
    # args.get_logger().info("Averaging client parameters")
    # parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]
    # new_nn_params = average_nn_parameters(parameters)
    
    
    
    ## Updating and Training the All Clients
    trainAccuracyOfAll=[]
    testAccuracyOfAll=[]
    testRecallOfAll=[]
    for client in clients:
        client_idx=0
        # args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
        # client.update_nn_parameters(new_nn_params)
        acc_train,_=client.train(epoch,train_data_loaders[client_idx])
        if epoch==KWARGS["ROUNDS"]:
            acc_test,_,_,recall_test=client.test()
        else:
            acc_test,_,_,recall_test=0,0,0,[0]
        trainAccuracyOfAll.append(acc_train)
        testAccuracyOfAll.append(acc_test)
        testRecallOfAll.append(recall_test)
        client_idx+=1
    return [trainAccuracyOfRand,trainAccuracyOfAll,testAccuracyOfAll,testRecallOfAll]

def create_clients(args, test_data_loader):
    """
    Create a set of clients.
    """
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(Client(args, idx, test_data_loader))

    return clients

def run_machine_learning(clients, args,KWARGS,distributed_train_dataset):
    """
    Complete machine learning over a series of clients.
    """
    overall_data=[]
    for epoch in range(1, args.get_num_epochs() + 1):
        round_data= train_subset_of_clients(epoch, args, clients,KWARGS,distributed_train_dataset)
        overall_data.append(round_data)
    csv_gen(KWARGS,overall_data)
def run_exp(KWARGS, client_selection_strategy, idx):
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(idx, 1)

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)

    args = Arguments(logger)
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(KWARGS["NUM_POISONED_WORKERS"])
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.log()
    args.set_num_epochs(KWARGS["ROUNDS"])
    args.set_num_workers(KWARGS["NUM_WORKERS"])
    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)
    KWARGS["EXPID"]=idx
    # Distribute batches equal volume IID
    distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())
    
    # distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)
    # distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers)
    # train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())

    clients = create_clients(args, test_data_loader)

    run_machine_learning(clients, args,KWARGS,distributed_train_dataset)
    # save_results(results, results_files[0])
    # save_results(worker_selection, worker_selections_files[0])

    logger.remove(handler)
