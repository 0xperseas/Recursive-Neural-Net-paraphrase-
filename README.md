# Recursive-Neural-Net-paraphrase-
A recursive neural net for paraphrase detection. There's also a Cky beam search parser at the bottom.


REFERENCE PAPER: https://nlp.stanford.edu/pubs/SocherLinNgManning_ICML2011.pdf

PREREQS: - import glove word vectors of 100 dimentions (file:glove.6B.100d.txt) from https://nlp.stanford.edu/projects/glove/
         - get parsed trees from stanford parser (*** they should be in binarized format ***)

DESCRIPTION: This code parses sentences already Annotated with stanford parser using binarization, and creates
a tree object for each sentence with Glove words embeddings for each leaf node (which is a word). Then it trains 
the objects on a recursive neural Tensor network build with tensorflow, and tries to predict the correct tree
labels found from the stanford parsed tree. It uses a bottom up algorithm to create the recursive tree and tries 
structure prediction on each node. Finally after each node's embeddings of the sentence has been trained, we could
construct a distance algorithm to compute the distances between sentences and get the ones that are close enough 
as paraphrases! 

EXTENSION: There's also a bottom up beam search Parser algorithm on the bottom of the page for fun. (Note: this would 
require a binarized format grammar.)
