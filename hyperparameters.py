import argparse
def parse_arguments():
    parser=argparse.ArgumentParser(description="Hyperparameter configuration", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_val", type=int, default=500, help="Number of validation nodes")
    parser.add_argument("--num_test", type=int, default=800, help="Number of test nodes")
    parser.add_argument("--num_coreset", type=int, default=None, help="Number of nodes for coreset selection")
    parser.add_argument("--K", type=int, default=20, help="Label budget")
       
    parser.add_argument("--radium", type=float, default=0.005, help="Radius for coreset selection")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha in HOIK")
    parser.add_argument("--beta", type=float, default=0.5, help="Injection accuracy parameter")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma for weighting diversity parameters")
    parser.add_argument("--prop_layer", type=int, default=1, help="Number of propagation layers")
    parser.add_argument("--use_propagation", type=str, default="HOIK", help="Propagation method to use")
    parser.add_argument("--use_dataset", type=str, default="Citeseer", help="Dataset to use")
    
    parser.add_argument("--degV", default=None, help="Vertex degree (internal use)")
    parser.add_argument("--degE", default=None, help="Edge degree (internal use)")
    parser.add_argument("--V", default=None, help="Vertex indices (internal use)")
    parser.add_argument("--E", default=None, help="Edge indices (internal use)")
    
    parser.add_argument("--dropout", default=0.6, type=float, help="Dropout rate")
    parser.add_argument("--hidden_size", default=64, type=int, help="Hidden layer size")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="Weight decay for optimizer")
    parser.add_argument("--train_turns", default=60, type=int, help="Number of training turns")

    args=parser.parse_args()

    return args

