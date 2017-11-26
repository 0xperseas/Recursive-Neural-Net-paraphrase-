import tensorflow as tf
import re
import queue
import numpy as np
import pickle
import random
from sklearn.preprocessing import OneHotEncoder
from queue import Queue, PriorityQueue
import os
import os.path
from collections import OrderedDict
from parse_glove import get_glove
from nltk.tokenize import sent_tokenize
import copy

#########################################################
############# Classes - Node, Pair  #####################
#########################################################

class Node:

    def __init__(self, label = None, word=None):
        self.label = label
        self.word = word
        self.parent = None
        self.left = None
        self.right = None
        self.index = None
        self.isLeaf = False
        self.vector = None
        self.score = None

    def __str__(self):
        if self == None:
            return
        return '({0} {1}, {2})'.format(str(self.label),
                                       str(self.left),
                                       str(self.right))

    def __repr__(self):
        if self == None:
            return
        return 'Node({0},({1}, {2}), {3})'.format(repr(self.label),
                                                  repr(self.left),
                                                  repr(self.right),
                                                  repr(self.parent))

class Pair(object):

    def __init__(self, first, second):
        self.first = first
        self.second = second

    def __repr__(self):
        return "Pair({0}, {1})".format(repr(self.first),
                                       repr(self.second))

    def __str__(self):
        s = "(" + str(self.first)
        second = self.second
        while isinstance(second, Pair):
            s += " " + str(second.first)
            second = second.second
        if second is not None:
            s += " . " + str(second)
        return s + ")"

class PriorityQueue(Queue):

    def peek(self):
        return self.queue[0]

    def remove(self, node):
        self.queue.remove(node)


############################ TODO ################################
# Penn treebank labels
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.8216&rep=rep1&type=pdf
# https://gist.github.com/nlothian/9240750 == LAbels parser
# binarize tree : https://github.com/stanfordnlp/treelstm/blob/master/lib/ConstituencyParse.java
# get good sentences - preprocess text if found in dictionary
# import word embeddings from 6B wikipedia if found in sentences
# stanford parser parse sentences  ==== ./lexparser.sh data/testsent.txt
# get annotations create classes of labels
# parse the sentences and create new trees data structures
# write function for correct nodes score total
# CYK parser bottom up beam search
# Max margin loss
# paraphrase detection nearest neighbors
# do it with recursive autoencoder

####### Import word vectors data glove ##########
glove_dictionary, glove_ids_to_words, \
glove_words_to_ids, glove_embed_matrix = get_glove()


trees_test = []

with open ('parsed_sents_paraphrase.txt', 'r') as f:
    for line in f:
        tree = line.rstrip()
        tree = tree.replace('(', ' ( ')
        tree = tree.replace(')', ' ) ')
        tree = tree.split()
        trees_test.append(tree)



########## Build the trees - Tree parser ############

def analyze_token(token):
    """Return the value of token if it can be analyzed
        as a number, or token."""
    try:
        return int(token)
    except (TypeError, ValueError):
        if '@' in token:
            token = token.replace('@', "")
            return token
        else:
            return token

def parse_tree(tree, children_list):

    token = analyze_token(tree.pop(0))
    if token != '(':
        return token
    else:
        token = tree.pop(0)
        if token != ')':
            #print(token)
            node = Node(token)
            result = traverse_parse_tree(tree,children_list)
            if len(result) == 2:
                node.left = result[0]
                node.right = result[1]
                node.label = analyze_token(token)
                node.left.parent = node
                node.right.parent = node
            else:
                node.label = analyze_token(token)
                node.word = str(result[0])
                #print(node.word)
                children_list.append(node)
            return node

def traverse_parse_tree(tree, children_list):
    children = []
    while tree[0] != ')':
        # tree.pop(0)
        children.append(parse_tree(tree, children_list))
    tree.pop(0)
    return children


#parsed_tree = parse_tree(trees_test[0], children_list)

counter =0
parsed_trees = []
grammar_dict = {}
children_grammar_dict = {}
map_children_list = {}

for tree in trees_test:
    children_list = []
    #print(tree)
    try:
        p_tree = parse_tree(tree, children_list)
        for x in children_list:
            x.score = random.uniform(0,1)
            if x.label in grammar_dict:
                if x.word not in grammar_dict[x.label]:
                    grammar_dict[x.label].append(x.word)
            else:
                grammar_dict[x.label] = [x.word]

            if x.word in children_grammar_dict:
                if x.label not in children_grammar_dict[x.word]:
                    children_grammar_dict[x.word].append(x.label)
            else:
                children_grammar_dict[x.word] = [x.label]

        #print(p_tree)
    except ValueError as e:
        print("Invalid format to parse treee!", e)
        continue
    parsed_trees.append(p_tree)
    map_children_list[counter] = children_list
    if counter % 500 == 0:
        print("Parsed ... {0} trees so far! ".format( counter))
    counter +=1

print("Parsed total Trees: ", len(parsed_trees))



def traverse_phrases(root, dict_to_label, update):

    if root != None:
        if root.left == None and root.right == None:
            if root.word.lower() in glove_dictionary :
                root.vector = glove_dictionary[root.word.lower()]
            else:
                vector = np.random.uniform(-1,1, [1,100])
                root.vector = vector
                glove_dictionary[root.word.lower()] = vector
                glove_embed_matrix = update['embed_matrix']
                glove_embed_matrix = np.append(glove_embed_matrix, vector, axis=0)
                update['embed_matrix'] = glove_embed_matrix
                glove_ids_to_words[len(glove_dictionary)-1] = root.word.lower()
                glove_words_to_ids[root.word.lower()] = len(glove_dictionary)-1

            root.isLeaf = True

        else:
            if root.label not in grammar_dict:
                grammar_dict[root.label] = [(root.left.label,
                                             root.right.label)]
            else:
                const = (root.left.label, root.right.label)
                if const not in grammar_dict[root.label]:
                    grammar_dict[root.label].append((root.left.label,
                                                     root.right.label))

            children = (root.left.label,root.right.label)

            if children not in children_grammar_dict:
                children_grammar_dict[children] = [root.label]
            else:
                if root.label not in children_grammar_dict[children]:
                    children_grammar_dict[children].append(root.label)

    if root == None:
        return
    left = traverse_phrases(root.left, dict_to_label, update)
    right = traverse_phrases(root.right, dict_to_label, update)
    dict_to_label[root.word] = root.label
    if root.parent != None:
        if root.parent.word != None:
            root.parent.word += " "+ root.word
        else:
            root.parent.word = root.word


#dict_to_label={}
#traverse_phrases(parsed_tree, dict_to_label, {})

t_trees = []
update= {}
update['embed_matrix'] = glove_embed_matrix
counter =0
for t_tree in parsed_trees:
    dict_to_label = {}
    try:
        traverse_phrases(t_tree, dict_to_label, update)
    except:
        print("Cannot traverse phrase -- Invalid format! \n",t_tree)
        continue
    t_trees.append(t_tree)
    if counter % 100 == 0:
        print("Traversed ... {0} trees so far! ".format( counter))
    counter +=1

print("Traversed Total ... {0} trees so far! ".format( counter))


glove_embed_matrix = update['embed_matrix']



def build_correct_tree(node,correc_tree_map):
    if node == None: return
    if type(node) is not list:
        if node.left.index != None and node.right.index != None:
            #print(node.left.index, node.right.index)
            node.index = (node.left.index[0], node.right.index[1])
            correc_tree_map.add(node.index)
            #print(node.index)
    if type(node) is list:
        for x in node:
            build_correct_tree(x.parent, correc_tree_map)
    else:
        build_correct_tree(node.parent,correc_tree_map)


index = 0
correct_trees_map = {}
for k, v in map_children_list.items():
    counter = 0
    correct_tree_map = set()
    for x in v:
        x.index = (counter, counter+1)
        correct_tree_map.add(x.index)
        counter +=1
    build_correct_tree(v, correct_tree_map)
    correct_trees_map[index] = correct_tree_map
    print("Building correect Trees ... traversed {0} trees so far! ".format( index))
    index +=1



#raise ValueError("")


#################### NEURAL NET Class ####################

class Model():

    def init_vars(self):
        embed_size = 100
        label_size = 1
        index_tuple_size = 2
        with tf.variable_scope('Embeddings'):
            tf.get_variable('embeddings', [len(glove_dictionary), embed_size],
                            dtype=tf.float64)

        self.children_index = tf.placeholder(tf.int32, shape= (None, index_tuple_size),
                                             name= 'children_index')
        self.correct_labels = tf.placeholder( tf.int32, shape=(None,index_tuple_size),
                                             name='correct_labels' )

        with tf.variable_scope('Composition'):
            tf.get_variable('V', [2* embed_size, 2* embed_size, embed_size],
                            dtype=tf.float64)
            tf.get_variable('W1', [2 * embed_size, embed_size],
                            initializer= tf.random_uniform_initializer(-1.0,1.0),
                            dtype=tf.float64)
            tf.get_variable('b1', [1, embed_size],
                            initializer =tf.constant_initializer(1.0),
                            dtype=tf.float64)
        with tf.variable_scope('Projection'):
            tf.get_variable('U', [embed_size, label_size],
                            initializer= tf.random_uniform_initializer(-1.0,1.0),
                            dtype=tf.float64)
            tf.get_variable('bs', [1, label_size],
                            initializer =tf.constant_initializer(1.0),
                            dtype=tf.float64)

        saver = tf.train.Saver()
        return saver


    def get_embeddings(self, word):
        with tf.variable_scope('Embeddings', reuse=True):
            embeddings = tf.get_variable('embeddings', dtype=tf.float64)
            tf.summary.scalar('embeddings', embeddings)
        return tf.expand_dims(tf.nn.embedding_lookup(
                        embeddings, glove_words_to_ids[word.lower()]), 0)


    def get_correct_tree(self, node, correc_tree_map ,total_score):

        with tf.variable_scope('Composition', reuse=True):
            V = tf.get_variable('V',   dtype=tf.float64)
            W1 = tf.get_variable('W1', dtype=tf.float64)
            b1 = tf.get_variable('b1', dtype=tf.float64)
            tf.summary.scalar('W1', W1)
            tf.summary.scalar('b1', b1)

        if node == None: return
        if type(node) is not list:
            if node.left.index != None and node.right.index != None:

                node.index = (node.left.index[0], node.right.index[1])
                correc_tree_map.add(node.index)
                if tf.contrib.framework.is_tensor(node.left.vector) and \
                   tf.contrib.framework.is_tensor(node.right.vector):
                    node_input = tf.concat([node.left.vector, node.right.vector], 1)
                    transpose_node_input = tf.transpose(
                                tf.concat([node.left.vector,  node.right.vector], 1))

                    node.vector = tf.tanh(tf.matmul(
                                        tf.matmul(node_input, V[:200,:200,0]),
                                        transpose_node_input) +
                                        tf.matmul(node_input, W1) + b1)

                    parent_score = tf.reduce_sum(self.get_projection(node.vector))
                    total_score.append(parent_score)

        if type(node) is list:
            for x in node:
                x.score = tf.reduce_sum(self.get_embeddings(x.word))
                x.vector = self.get_embeddings(x.word)
                total_score.append(x.score)
                self.get_correct_tree(x.parent,
                        correct_tree_map, total_score)
        else:
            self.get_correct_tree(node.parent,
                        correct_tree_map, total_score)
        return total_score


    def build_tree(self, children, edges_map, order_scores, build_children= False):
        # q.pop()
        # if child is leaf then get embedding
        # score of leaves is tanh(sum(embedding))
        # else get the siblings vectors from q(x.vector)
        # concat nodes vectors
        # create parent Node() and vector
        # add parent to the queue
        # sum the scores of the tree
        # in order to get the num incorrect nodes
        # get all the graph (x, i, j) in the end
        # traverse the correct tree to build a graph dict [X,1,2] = Node
        # compare if current node in the graph dict correct
        # return ordered_dict and sum score of tree
        # compute max margin (sum scores tree + D(n)) - current_max(tree)
        # Store all the trees and compare current best with the other
        ###### Get the names of the graph #######
        # tf.get_default_graph().as_graph_def()

        with tf.variable_scope('Composition', reuse=True):
            V = tf.get_variable('V',   dtype=tf.float64)
            W1 = tf.get_variable('W1', dtype=tf.float64)
            b1 = tf.get_variable('b1', dtype=tf.float64)
            tf.summary.scalar('W1', W1)
            tf.summary.scalar('b1', b1)

        order_dict = OrderedDict()

        if build_children:
            for i in range(len(children)):
                children[i].score = tf.reduce_sum(self.get_embeddings(
                                                 children[i].word))
                children[i].vector = self.get_embeddings(children[i].word)
                edges_map[children[i].index] = [children[i].score,
                                                children[i].vector,
                                                children[i].index]
                order_scores[children[i].index] = children[i].score

            return order_scores, children
        else:

            # do concat vectors and build parent
            c1_vector = children[0][1]
            c2_vector = children[1][1]
            c1_index = children[0][0]
            c2_index = children[1][0]
            c1_score = order_scores[(c1_index[0], c1_index[1])]
            c2_score = order_scores[(c2_index[0], c2_index[1])]

            node_input = tf.concat([c1_vector, c2_vector], 1)
            transpose_node_input = tf.transpose(
                                    tf.concat([c1_vector,  c2_vector], 1))

            parent_vector = tf.tanh(tf.matmul(
                                tf.matmul(node_input, V[:200,:200,0]),
                                transpose_node_input) +
                                tf.matmul(node_input, W1) + b1)

            parent_score = tf.reduce_sum(self.get_projection(parent_vector))
            parent_index = self.build_edges(c1_index, c2_index)
            parent = [parent_score, parent_vector, parent_index]
            edges_map[parent_index] = parent
            order_scores[parent_index] = parent_score

            return order_scores, parent

    def get_siblings(self, node_index, edges_map, sess):
        sibling1 = None
        sibling2 = None
        for k, v in edges_map.items():
            if node_index[0] == k[1]:
                sibling1 = v
            elif node_index[1] == k[0]:
                sibling2 = v
        if sibling1 != None and sibling2 != None:

            if sess.run(sibling1[0]) > sess.run(sibling2[0]):
                return sibling1
            elif sess.run(sibling1[0]) < sess.run(sibling2[0]):
                return sibling2
            else:
                res = random.randint(0,1)
                if res == 0:
                    return sibling1
                else:
                    return sibling2
        elif sibling1 != None:
            return sibling1
        elif sibling2 != None:
            return sibling2
        else:
            return None



    def build_edges(self, c1_index, c2_index):
        if c1_index[1] == c2_index[0]:
            return (c1_index[0], c2_index[1])
        elif c1_index[0] == c2_index[1]:
            return (c2_index[0], c1_index[1])


    def get_projection(self, parent_vector):
        with tf.variable_scope('Projection', reuse=True):
            U = tf.get_variable('U', dtype=tf.float64)
            bs = tf.get_variable('bs', dtype=tf.float64)
            tf.summary.scalar('U', U)
            tf.summary.scalar('bs', bs)
#           score = tf.matmul(parent_vector, U) + bs
            score = tf.matmul(parent_vector, U)
        return score



    def get_incorrect_trees(self,ordered_tree, sent_index, sess):

        counter = 0
        for k, v in ordered_tree.items():
            print(k, sess.run(v))
            if k not in correct_trees_map[sent_index]:
                counter+=1
        print("GET CORRECT _TREE", correct_trees_map[sent_index])
        print("NUMBER OF INCORRECT TREES: ", counter)
        return counter

    def get_loss(self, ordered_tree, sess,sent_index,
                 correct_tree_score, max_score=None):
        scores=[]
        for k, v in ordered_tree.items():
            scores.append(v)
        num_incorrect_trees = self.get_incorrect_trees(ordered_tree,
                                                       sent_index, sess)
        if max_score == None:
            loss =  (tf.reduce_sum(scores) + \
                    tf.multiply(tf.cast(num_incorrect_trees, tf.float64), 0.1)) \
                    - correct_tree_score

            return loss, tf.reduce_sum(scores), num_incorrect_trees
        else:
            loss = tf.abs((tf.reduce_sum(scores) + \
                tf.multiply(tf.cast(num_incorrect_trees,
                                    tf.float64), 0.1)) - max_score)

            with tf.variable_scope('Composition', reuse=True):
                W1 = tf.get_variable('W1', dtype=tf.float64)
                V = tf.get_variable('V', dtype=tf.float64)
                tf.summary.scalar('W1', W1)
            with tf.variable_scope('Projection', reuse=True):
                U = tf.get_variable('U', dtype=tf.float64)
                tf.summary.scalar('U', U)

            return loss + 0.02 * (tf.nn.l2_loss(W1) +
                                    tf.nn.l2_loss(U) +
                                    tf.nn.l2_loss(V)), \
                                    tf.reduce_sum(scores), \
                                    num_incorrect_trees


    def train(self, loss):
        return tf.train.GradientDescentOptimizer(0.0001).minimize(loss)





#print("True grammar:",grammar_dict)
#print("Children Grammar:", children_grammar_dict)

num_childs =0
for val in children_grammar_dict.values():
    if len(val) > 1:
        num_childs +=1

def get_new_children(children):
    new_children = []
    index_counter = 0
    sentence = ""
    for node in children:
        new_node = Node()
        new_node.isLeaf = True
        new_node.word = node.word
        new_node.label = node.label
        new_node.score = random.uniform(0,1)
        new_node.vector = node.vector
        new_node.index = (index_counter, index_counter+1)
        index_counter +=1
        sentence += " " +new_node.word
        new_children.append(new_node)
    print("SENTENCE: ", sentence)
    return new_children



def run_neural_net():

    with tf.Session() as sess:
        epochs = 10
        rnn = Model()
        saver = rnn.init_vars()
        init = tf.global_variables_initializer()
        sess.run(init)

        for k_index ,child_vals in map_children_list.items():
            max_score = 0
            correct_tree_score = 0
            min_incorrect_trees =0

            for epoch in range(epochs):
                temp_children = copy.deepcopy(child_vals)
                new_children = get_new_children(temp_children)
                order_scores = OrderedDict()
                edges_map = {}
                nodes_visited = set()
                q = queue.PriorityQueue()
                correct_score = rnn.get_correct_tree(temp_children, {},[])
                correct_tree_score = sum(sess.run(correct_score))
                ordered_scores, q_children = rnn.build_tree(new_children,
                                                            edges_map,
                                                            order_scores, True)
                for x in q_children:
                    score = sess.run(x.score)
                    q.put((-1 * score, x.index))

                loss = 0
                while q.qsize() > 0:
                    node = q.get()
                    node_score = node[0]
                    node_index = node[1]
                    if node_index in nodes_visited:
                        continue
                    node_vector = edges_map[node_index][1]
                    print("NEW NODE SCORE:",node_score, node_index)
                    sibling_node = rnn.get_siblings(node_index, edges_map, sess)
                    nodes_visited.add(node_index)
                    if sibling_node != None:
                        sibling_score= sibling_node[0]
                        sibling_vector = sibling_node[1]
                        sibling_index = sibling_node[2]
                        nodes_visited.add(sibling_index)
                        edges_map.pop(sibling_index, None)
                        edges_map.pop(node_index, None)
                        children  = [[node_index, node_vector],
                                     [sibling_index, sibling_vector]]
                        ordered_scores, parent = rnn.build_tree(children, edges_map,
                                                                order_scores, False)
                        q.put((-1 * sess.run(parent[0]), parent[2]))

#                        loss, scores, num_incorrect_trees = rnn.get_loss(ordered_scores,
#                                                                     sess, k_index,
#                                                                    correct_tree_score)
#                        train_op = rnn.train(loss)
#                        loss_val, _ , scores= sess.run([loss, train_op, scores])
#                        print("LOSS: ",loss_val)
#                        print("NEW SCORE: ", scores)
#                        print("CORRECT SCORE!!!:", correct_tree_score)


                loss, scores, num_incorrect_trees = rnn.get_loss(ordered_scores,
                                                             sess, k_index,
                                                            correct_tree_score)
                train_op = rnn.train(loss)
                loss_val, _ , scores= sess.run([loss, train_op, scores])
                print("LOSS: ",loss_val)
                print("NEW SCORE: ", scores)
                print("CORRECT SCORE!!!:", correct_tree_score)

                ######################### Latent SGD ###############################
#                if epoch == 0:
#                    loss, scores, num_incorrect_trees = rnn.get_loss(ordered_scores,
#                                                                     sess, k_index)
#                    train_op = rnn.train(loss)
#                    loss_val, _ , scores= sess.run([ loss, train_op, scores])
#                    max_score = scores
#                    min_incorrect_trees = num_incorrect_trees
#                    print("LOSS: ",loss_val)
#                    print("SCORES: ", scores)
#                else:
#                    loss, scores, num_incorrect_trees = rnn.get_loss(ordered_scores,
#                                                                     sess,k_index, max_score)
#                    train_op = rnn.train(loss)
#                    loss_val, _ , scores= sess.run([ loss, train_op, scores])
#                    print("SCORES: ", max_score, scores)
##                    if num_incorrect_trees < min_incorrect_trees:
##                        max_score = scores
#                    if max_score < scores:
#                        max_score = scores
#                    print("LOSS: ",loss_val)

                merged = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(
                    '/Users/perseaschristou/ai/ml/tensorboard/custom_test', sess.graph)
                #summary, loss_val, _ , scores= sess.run([merged, loss, train_op, scores])
                #train_writer.add_summary(summary, i)
                print("\n############################################################################\n\n\n")



run_neural_net()

print(len(map_children_list), len(correct_trees_map))





#######################################################################################################
########################  C K Y  B E A M   S E A R C H  P A R S E R  ##################################
#######################################################################################################
edges_map = {}
chart = np.ones((len(grammar_dict), len(new_children), len(new_children)+1))
grammar_to_index = {}
ids =0
for k, v in grammar_dict.items():
    grammar_to_index[k] = ids
    ids+=1

def get_parents_helper(const, sibling, v, beg, end, side):
    parents = []
    siblings = []
    for k in edges_map.keys():
        if sibling == edges_map[k]['label'][0]:
            sibling_index = edges_map[k]['label']
            if sibling_index[1] == end and side == 0:
                sibling_0 = sibling_index[0]
                sibling_1 = sibling_index[1]
                sibling_2 = sibling_index[2]
                parent_label = v
                parent_index = (parent_label, beg, sibling_2)
                parents.append(parent_index)
                siblings.append((sibling_0, sibling_1, sibling_2))

                print("---------- L E F T ---------------")
                print("PARENT INDEX",(parent_label,parent_index))
                print("SIBLING INDEX:",(sibling_0, sibling_1, sibling_2))
                print("CONST INDEX ", (const[1][0], beg, end))

            elif sibling_index[2] == beg and side == 1:
                sibling_0 = sibling_index[0]
                sibling_1 = sibling_index[1]
                sibling_2 = sibling_index[2]
                parent_label = v
                parent_index = (parent_label, sibling_1 , end)
                parents.append(parent_index)
                siblings.append((sibling_0, sibling_1, sibling_2))

                print("---------- R I G H T -------------")
                print("PARENT INDEX",(parent_label,parent_index))
                print("SIBLING INDEX:",(sibling_0, sibling_1, sibling_2))
                print("CONST INDEX ", (const[1][0], beg, end))

    return parents, siblings


def get_parents(const, beg,end):
    parents_siblings_pair = []
    for k, vals in children_grammar_dict.items():
        #print(const[1], k)
        if type(k) == tuple:
            for v in vals:
                if const[1][0] == k[0]:
                    sibling = k[1]
                    parents, siblings = get_parents_helper(const,
                                            sibling, v, beg,end,0)
                    if len(parents) > 0:
                        parents_siblings_pair.append((parents[0],
                                                      siblings[0]))
                elif const[1][0] == k[1]:
                    sibling = k[0]
                    parents, siblings = get_parents_helper(const,
                                            sibling, v, beg,end,1)
                    if len(parents) > 0:
                        parents_siblings_pair.append((parents[0],
                                                      siblings[0]))
    return parents_siblings_pair


def update_chart(const, parent, sibling):
    edges_map[parent] = {}
    edges_map[parent]['label'] = parent
    edges_map[parent]['backtrack'] = (const[1], sibling)

    print("SIBLING SCORE:",chart[grammar_to_index[sibling[0]],
                                 sibling[1], sibling[2]])
    print("CONSTIT SCORE:", chart[grammar_to_index[const[1][0]],
                                  const[1][1], const[1][2]])
    print("PARENT SCORE:",chart[grammar_to_index[parent[0]],
                                parent[1], parent[2]])

    new_score = chart[grammar_to_index[const[1][0]],
                      const[1][1], const[1][2]]+ \
                chart[grammar_to_index[sibling[0]],
                      sibling[1], sibling[2]]
                # + chart[grammar_to_index[parent[0]], parent[1], parent[2]]

    if chart[grammar_to_index[parent[0]], parent[1],
             parent[2]] > new_score:
        edges_map[parent]['score'] = new_score
        chart[grammar_to_index[parent[0]], parent[1],
              parent[2]] = new_score
        return True
    return False


####################### TODO - CORRECT ##########################
# While pq is not empty                                         #
# Create all triplets on edges-map                              #
# check for many expands on each label                          #
# that's why we need triplets. (X, i, j)                        #
# as  keys to the map                                           #
# Keep k-best subtrees for test                                 #
# Should use a chart table?                                     #
#---------------------------------------------------------------#
# STORE backpointers to parents                                 #
#                                                               #
# Create beam size                                              #
# !!!!!for each possible parent in child_gram_map!!!!!!         #
#################################################################

def bottom_up_beam_search():

    q = PriorityQueue()

    for i in range(len(new_children)):
            indexes = (new_children[i].label,i, i+1)
            edges_map[indexes] = {}
            edges_map[indexes]['label'] = (new_children[i].label, i, i+1)
            edges_map[indexes]['score'] = new_children[i].score
            edges_map[indexes]['backtrack'] = (new_children[i].word)
            q.put((edges_map[indexes]['score'] , edges_map[indexes]['label']))
            chart[grammar_to_index[new_children[i].label], i, i+1] = new_children[i].score


    counter = 0
    while not q.empty():
        const = q.get()
        print(const)
        const_index = const[1]
        parents_siblings = get_parents(const, const_index[1],
                                                const_index[2])

        print("RETURN BEAMS:",parents_siblings)
        if len(parents_siblings) >0:
            for i in range(len(parents_siblings)):  # beam size 1
                                                    #(should get the best const though)
                if update_chart(const, parents_siblings[i][0],
                                       parents_siblings[i][1]):

                    q.put((edges_map[parents_siblings[i][0]]['score'],
                           edges_map[parents_siblings[i][0]]['label']))

                    print("\n\n\n ADDED TO THE QUEUE!!! \n\n\n")
        counter +=1
        print("######################################################### ", counter)


    print([(x.word ,x.label) for x in children_list])
    print("Number with more than one values in grammar children:", num_childs)

    print([(x , y['backtrack']) for x, y in edges_map.items()])

#bottom_up_beam_search()




















