/*
 * main.cpp
 * 
 * Created by: Ashik <ashik@KAI10>
 * Created on: Mon, 17 Apr 2017
 */


#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

#define mem(list, val) memset(list, (val), sizeof(list))
#define pb push_back
#define N_ITERS 1000
#define N_BURN_IN 500
#define STRIDE 50

typedef vector<vector<int>> Matrix;
typedef vector<vector<double>> dMatrix;

vector<string> corpus, voc;
unordered_set<string> vocabulary;

unordered_map<string, int> index_in_voc;

vector<int> docStart, Z, W;

int K, D, V, N; 
double alpha, eta;

Matrix saveZ;

#include "utilities.hpp"

void LDA(vector<int> &Z, Matrix &n_doc_topic, Matrix &n_topic_voc)
{
    vector<vector<int>> curResult;
    dMatrix beta(K, vector<double>(V, 0));

    vector<double> P(K, 0.0);

    vector<int> n_doc_topic_sum(D, 0), n_topic_voc_sum(K, 0);
    for(int d=0; d<D; d++){
        for(int t=0; t<K; t++) n_doc_topic_sum[d] += n_doc_topic[d][t];
    }

    for(int t=0; t<K; t++){
        for(int v=0; v<V; v++) n_topic_voc_sum[t] += n_topic_voc[t][v];
    }

    //int stableCount = 0;

    for(int r=0; r<N_ITERS; r++){
        cout << "iteration: " << r << "\n";

        for(int i=0; i<N; i++){
            //cout << i << endl;
            string word = corpus[i];
            int topic = Z[i];
            int doc = W[i];
            int vIndex = index_in_voc[word];

            //excluding i'th word
            n_doc_topic[doc][topic]--;
            n_topic_voc[topic][vIndex]--;

            n_doc_topic_sum[doc]--;
            n_topic_voc_sum[topic]--;

            for(int t=0; t<K; t++){
                
                int n_doc_i_topic = n_doc_topic[doc][t];
                int n_i_topic_voc = n_topic_voc[t][vIndex];

                int alphaSum = 0, etaSum = 0;

                //for(int k=0; k<K; k++) alphaSum += n_doc_topic[doc][k];
                //for(int v=0; v<V; v++) etaSum += n_topic_voc[t][v];
                alphaSum = n_doc_topic_sum[doc];
                etaSum = n_topic_voc_sum[t];

                P[t] = (alpha + n_doc_i_topic) * (eta + n_i_topic_voc) / ((K*alpha + alphaSum) * (V*eta + etaSum));
            }

            normalize(P);
            int newTopic = pickTopic(P);

            Z[i] = newTopic;
            n_doc_topic[doc][newTopic]++;
            n_topic_voc[newTopic][vIndex]++;

            n_doc_topic_sum[doc]++;
            n_topic_voc_sum[newTopic]++;
        }

        //burn in
        if(r < N_BURN_IN) continue;

        //stride
        if(r % STRIDE == 0) saveZ.push_back(Z);

        /*
        for(int t=0; t<K; t++){ //for each topic
            for(int w=0; w<V; w++){ //for each word
                int count = 0;
                for(int i=0; i<Z.size(); i++){
                    if(corpus[i] == voc[w] && Z[i] == t) count++;
                }
                beta[t][w] += 1.0*count;
            }
        }

        //for(auto &v: beta) normalize(v);

        vector<vector<int>> newResult;
        report(beta, newResult);

        //cout << "newResult:\n";
        //showResult(newResult);
        
        //getchar();

        if(r == 0){
            curResult = newResult;
            continue;
        }

        bool converged = true;
        for(int i=0; i<curResult.size(); i++){
            if(!equal(curResult[i], newResult[i])){
                converged = false;
                break;
            }
        }
        if(converged){
            stableCount++;

            if(stableCount >= 300){
                cout << "converged :)\n at iteration: " << r << endl;
                showResult(curResult);
                break;
            }
        }
        else stableCount = 0;

        cout << "stableCount: " << stableCount << endl;
        
        
        curResult = newResult;
        */
    }

    //constructing theta
    /*
    dMatrix theta(D, vector<double>(K, 0));
    
    for(int d=0; d<D; d++){ //for each document
        for(int t=0; t<K; t++){ //for each topic
            int count = 0;
            for(auto z: saveZ){ //for each state of Z
                for(int i=docStart[d]; W[i] == d && i<N; i++){
                    if(z[i] == t) count++;
                }
            }
            theta[d][t] = 1.0*count;
        }
    }
    
    for(auto &v: theta){
        normalize(v);
        for(auto p: v) cout << p << ' ';
        cout << endl;
    }
    */

    
    //constructing beta
    for(int t=0; t<K; t++){ //for each topic
        for(int w=0; w<V; w++){ //for each word
            int count = 0;
            for(auto z: saveZ){
                for(int i=0; i<N; i++){
                    if(corpus[i] == voc[w] && z[i] == t) count++;
                }
            }
            beta[t][w] += 1.0*count;
        }
    }

    for(auto &v: beta) normalize(v);
    report(beta, curResult);
    
    showResult(curResult);
    
}

int main(int argc, char **argv)
{
    //freopen("test.txt", "r", stdin);
    srand(time(NULL));

    if(argc < 3){
        puts("Usage: Number_of_Topics Number_of_Documents");
        exit(0);
    }

    K = atoi(argv[1]);
    D = atoi(argv[2]);

    readData(D, corpus, W, vocabulary);

    cout << "K: " << K << " D: " << D << " N: " << N << " V: " << V << '\n';

    Matrix n_doc_topic(D, vector<int>(K,0)), n_topic_voc(K, vector<int>(V, 0));
    initialize(Z, n_doc_topic, n_topic_voc);

    Matrix start_n_doc_topic = n_doc_topic, start_n_topic_voc = n_topic_voc;
    vector<int> startZ = Z;

    LDA(Z, n_doc_topic, n_topic_voc);
    /*
    string a[] = {"a", "b","d"}, b[] = {"a", "b","c"};
    vector<string> A(a, a+3), B(b, b+3);
    cout << equal(A, B) << endl;
    */

    return 0;
}
