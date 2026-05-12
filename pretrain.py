import argparse
import torch.optim as optim
from torch.utils.data import DataLoader

import rdkit
import sys
from tqdm import tqdm
from gnn_model import GNN
from decoder import Model_decoder

sys.path.append('./util/')

from data_utils import *
from motif_generation import *

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def group_node_rep1(node_rep, batch_size, num_part):
    group = []
    super_group = []
    count = 0
    for i in range(batch_size):
        num_atom = num_part[i][0]
        num_motif = num_part[i][1]
        num_all = num_atom + num_motif + 1
        group.append(node_rep[count:count + num_atom])
        super_group.append(node_rep[count + num_all - 1])
        count += num_all
    return group, super_group


def group_node_rep2(node_rep, batch_index, batch_size):
    group = []
    count = 0
    for i in range(batch_size):
        num = sum(batch_index == i)
        group.append(node_rep[count:count + num])
        count += num
    return group


def train(args, model_list, loader, optimizer_list, device):
    if args.ablation != 'w/o_motif':
        model, model_decoder, motif_model = model_list
        motif_model.train()
    else:
        model, model_decoder = model_list

    model.train()
    model_decoder.train()
    if_auc, if_ap, type_acc, a_type_acc, a_num_rmse, b_num_rmse, word_acc = 0, 0, 0, 0, 0, 0, 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch_size = len(batch)
        mol_graphs, mol_trees = zip(*batch)

        graph_batch1 = molgraph_to_graph_data(mol_graphs)
        graph_batch1 = graph_batch1.to(device)
        node_rep1 = model(graph_batch1.x, graph_batch1.edge_index, graph_batch1.edge_attr)
        num_part = graph_batch1.num_part
        node_rep1, super_node_rep = group_node_rep1(node_rep1, batch_size, num_part)

        loss1, bond_if_auc, bond_if_ap, bond_type_acc, atom_type_acc, atom_num_rmse, bond_num_rmse = model_decoder(
            mol_graphs, node_rep1, super_node_rep)

        if args.ablation != 'w/o_motif':
            graph_batch2 = moltree_to_graph_data(mol_trees)
            batch_index = graph_batch2.batch.numpy()
            graph_batch2 = graph_batch2.to(device)
            node_rep2 = model(graph_batch2.x, graph_batch2.edge_index, graph_batch2.edge_attr)
            node_rep2 = group_node_rep2(node_rep2, batch_index, batch_size)
            loss2, wacc, tacc = motif_model(mol_trees, node_rep2)

            loss = loss1 + loss2
            word_acc += wacc

        else:
            loss = loss1

        optimizer_list.zero_grad()

        loss.backward()

        optimizer_list.step()

        if_auc += bond_if_auc
        if_ap += bond_if_ap
        type_acc += bond_type_acc
        a_type_acc += atom_type_acc
        a_num_rmse += atom_num_rmse
        b_num_rmse += bond_num_rmse

        if (step + 1) % 20 == 0:
            if_auc = if_auc / 20
            if_ap = if_ap / 20
            type_acc = type_acc / 20
            a_type_acc = a_type_acc / 20
            a_num_rmse = a_num_rmse / 20
            b_num_rmse = b_num_rmse / 20
            word_acc = word_acc / 20 * 100

            print('Batch:', step, 'loss:', loss.item(), 'Word:', word_acc)
            if_auc, if_ap, type_acc, a_type_acc, a_num_rmse, b_num_rmse, word_acc = 0, 0, 0, 0, 0, 0, 0


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=512, help='embedding dimensions (default: 512)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined (last, sum, max or concat)')
    parser.add_argument('--dataset', type=str, default='./data/zinc/all.txt')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--output_model_file', type=str, default='./saved_model/pretrain_bfs.pth',
                        help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    parser.add_argument("--hidden_size", type=int, default=512, help='hidden size')
    parser.add_argument('--input_model_file', type=str, default='./saved_model/init')
    parser.add_argument("--latent_size", type=int, default=56, help='latent size')
    parser.add_argument("--vocab", type=str, default='./data/zinc/clique.txt', help='vocab path')
    parser.add_argument('--order', type=str, default="bfs", help='motif tree generation order (bfs or dfs)')
    parser.add_argument('--ablation', type=str, default="keep_all", help='keep_all, w/o_node, w/o_graph and w/o_motif')
    args = parser.parse_args()

    print(args)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dataset = MoleculeDataset(args.dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        collate_fn=lambda x: x, drop_last=True)

    model = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(
        device)
    model_decoder = Model_decoder(args.hidden_size, args.ablation, device).to(device)

    if args.ablation != 'w/o_motif':
        vocab = [x.strip("\r\n ") for x in open(args.vocab)]
        vocab = Vocab(vocab)
        motif_model = Motif_Generation(vocab, args.hidden_size, args.latent_size, 3, device, args.order).to(device)
        model_list = [model, model_decoder, motif_model]
        optimizer = optim.Adam([{"params": model.parameters()}, {"params": model_decoder.parameters()},
                                {"params": motif_model.parameters()}], lr=args.lr,
                               weight_decay=args.decay)

    else:
        model_list = [model, model_decoder]
        optimizer = optim.Adam([{"params": model.parameters()}, {"params": model_decoder.parameters()}], lr=args.lr,
                               weight_decay=args.decay)

    for epoch in range(1, args.epochs + 1):
        print('====epoch', epoch)
        train(args, model_list, loader, optimizer, device)

        if not args.output_model_file == "":
            torch.save(model.state_dict(), args.output_model_file)


if __name__ == "__main__":
    main()
