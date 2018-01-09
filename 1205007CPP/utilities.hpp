random_device rd;
uniform_real_distribution<double> rng(0.0, 1.0);

inline double getRandomDouble(){return rng(rd);}
inline int getRandomTopic(){return rand()%K;}
inline int getDocumentNumber(int wordIndex){return W[wordIndex];}

void readData(int documents, vector<string> &corpus, vector<int> &W, unordered_set<string> &vocabulary)  
{
	ifstream fin;
	string word;

	for(int i=0; i<documents; i++){
		fin.open("../20newsgroups/" + to_string(i+1));

		docStart.push_back(corpus.size());

		while(fin >> word){
			corpus.push_back(word);
			W.push_back(i);
			vocabulary.insert(word);
		} 

		fin.close();
	}

	V = vocabulary.size();
	N = corpus.size();

	for(auto w: vocabulary) voc.push_back(w);
	for(int i=0; i<V; i++) index_in_voc[voc[i]] = i;
}

void initialize(vector<int> &Z, Matrix &n_doc_topic, Matrix &n_topic_voc)
{
	alpha = 50.0/K;
    eta = 0.1;

    for(int i=0; i<N; i++) Z.push_back(getRandomTopic());

    for(int d=0; d<D; d++){ ///for each documant
    	for(int t=0; t<K; t++){ ///for each topic
    		int count = 0;
    		for(int i=docStart[d]; getDocumentNumber(i) == d && i<N; i++){
    			if(Z[i] == t) count++;
    		}
    		n_doc_topic[d][t] = count;
    	}
    }

    for(int t=0; t<K; t++){ //for each topic
    	for(int w=0; w<V; w++){ //for each word
    		int count = 0;
    		for(int i=0; i<N; i++){
    			if(corpus[i] == voc[w] && Z[i] == t) count++;
    		}
    		n_topic_voc[t][w] = count;
    	}
    }
}

void normalize(vector<double> &P)
{
	double sum = 0;
	for(auto v: P) sum += v;
	for(auto &v: P) v /= sum;
}

int pickTopic(vector<double> &P)
{
	vector<double> CP(P.size(), P[0]);
	for(auto i=1; i<CP.size(); i++) CP[i] = CP[i-1] + P[i];

	double num = getRandomDouble();
	auto it = lower_bound(CP.begin(), CP.end(), num);
	return (it - CP.begin());
}

void showResult(vector<vector<int>> &res)
{
	ofstream fout;
	fout.open("topicwords.csv");

    for(int i=0; i<res.size(); i++){
    	for(int j=0; j<res[i].size(); j++){
    		cout << voc[res[i][j]] << ' ';
    		if(j>0) fout << ',' << voc[res[i][j]];
    		else fout << voc[res[i][j]]; 
    	}
    	cout << endl;
    	fout << endl;
    }
    fout.close();
}

void report(dMatrix &beta, vector<vector<int>> &curResult)
{
    for(auto t=0; t<K; t++){
    	//cout << "\ntopic " << t << ":\n";
    	vector<pair<double, int>> res;

    	for(auto i=0; i<V; i++) res.push_back(make_pair(beta[t][i], i));
    	sort(res.rbegin(), res.rend());

    	vector<int> temp;

    	for(int i=0; i<5; i++){
    		//cout << res[i].second << ',' << res[i].first << endl;

    		temp.push_back(res[i].second);
    	}
    	curResult.push_back(temp);
    }

}

bool equal(vector<int> &p, vector<int> &q){
	sort(p.begin(), p.end());
	sort(q.begin(), q.end());

	vector<string> diff(p.size()+q.size());
	auto it = set_difference(p.begin(), p.end(), q.begin(), q.end(), diff.begin());
	diff.resize(it - diff.begin());
	return (diff.size() == 0); 
}
