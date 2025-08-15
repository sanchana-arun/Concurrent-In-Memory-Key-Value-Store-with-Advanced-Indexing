// Concurrent In-Memory Key-Value Store with Advanced Indexing
// Single-file C++17 implementation with:
//  - O(1) average lookups via striped hash table
//  - O(log n) range queries via ordered index (std::map)
//  - Thread-safety using per-bucket shared_mutex + global ordered-index shared_mutex
//  - Lock ordering to avoid deadlocks
//  - Minimal interactive shell for manual testing
//
// Instructions to run:
//  - Build:  g++ -std=gnu++17 -O2 -pthread kvstore.cpp -o kvstore
//  - Run:    ./kvstore


#include <bits/stdc++.h>
#include <shared_mutex>
#include <thread>
#include <atomic>

using namespace std;

//utilities
static inline size_t hash_key(const string &k) {
    return std::hash<string>{}(k);
}

//a striped, thread-safe hash index that maps key -> shared_ptr<value>
class StripedHashIndex {
public:
    struct Bucket {
        mutable std::shared_mutex mtx; //shared for reads, unique for writes
        unordered_map<string, shared_ptr<string>> map;
    };

    explicit StripedHashIndex(size_t stripes = 64) : buckets_(max<size_t>(1, stripes)) {}

    bool get(const string &key, shared_ptr<string> &out) const {
        auto &b = bucket_for(key);
        shared_lock<shared_mutex> g(b.mtx);
        auto it = b.map.find(key);
        if (it == b.map.end()) return false;
        out = it->second;
        return true;
    }

    void put(const string &key, const shared_ptr<string> &val) {
        auto &b = bucket_for(key);
        unique_lock<shared_mutex> g(b.mtx);
        b.map[key] = val;
    }

    bool erase(const string &key) {
        auto &b = bucket_for(key);
        unique_lock<shared_mutex> g(b.mtx);
        return b.map.erase(key) > 0;
    }

    size_t size_approx() const {
        size_t s = 0;
        for (auto &b : buckets_) {
            shared_lock<shared_mutex> g(b.mtx);
            s += b.map.size();
        }
        return s;
    }

    //acquire a unique lock for the bucket owning `key`.
    unique_lock<shared_mutex> lock_bucket_unique(const string &key) {
        auto &b = bucket_for(key);
        return unique_lock<shared_mutex>(b.mtx);
    }

    //acquire a shared lock for the bucket owning `key`.
    shared_lock<shared_mutex> lock_bucket_shared(const string &key) const {
        auto &b = bucket_for(key);
        return shared_lock<shared_mutex>(b.mtx);
    }

    //access to bucket map under external lock (use with care)
    unordered_map<string, shared_ptr<string>> &unsafe_bucket_map(const string &key) {
        return bucket_for(key).map;
    }

private:
    vector<Bucket> buckets_;

    Bucket &bucket_for(const string &key) const {
        size_t idx = hash_key(key) % buckets_.size();
        return const_cast<Bucket&>(buckets_[idx]);
    }
};

//ordered index: std::map with shared_mutex for concurrent readers
class OrderedIndex {
public:
    bool get(const string &key, shared_ptr<string> &out) const {
        shared_lock<shared_mutex> g(mtx_);
        auto it = map_.find(key);
        if (it == map_.end()) return false;
        out = it->second;
        return true;
    }

    void put(const string &key, const shared_ptr<string> &val) {
        unique_lock<shared_mutex> g(mtx_);
        map_[key] = val;
    }

    bool erase(const string &key) {
        unique_lock<shared_mutex> g(mtx_);
        return map_.erase(key) > 0;
    }

    vector<pair<string,string>> range(const string &lo, const string &hi, size_t limit = SIZE_MAX) const {
        vector<pair<string,string>> res;
        shared_lock<shared_mutex> g(mtx_);
        auto it = map_.lower_bound(lo);
        for (; it != map_.end() && it->first <= hi && res.size() < limit; ++it) {
            res.emplace_back(it->first, *it->second);
        }
        return res;
    }

    size_t size() const {
        shared_lock<shared_mutex> g(mtx_);
        return map_.size();
    }

    //expose locks for coordinated updates with hash index
    unique_lock<shared_mutex> lock_unique() { return unique_lock<shared_mutex>(mtx_); }
    shared_lock<shared_mutex> lock_shared() const { return shared_lock<shared_mutex>(mtx_); }

    //UNLOCKED ops — caller must hold the appropriate lock already.
    void put_unlocked(const string &key, const shared_ptr<string> &val) { map_[key] = val; }
    bool erase_unlocked(const string &key) { return map_.erase(key) > 0; }

private:
    mutable shared_mutex mtx_;
    map<string, shared_ptr<string>> map_;
};

//KVStore combining both indices with consistent updates.
class KVStore {
public:
    explicit KVStore(size_t stripes = 64) : hash_(stripes) {}

    //PUT: update both indices atomically under a well-defined lock order.
    void put(const string &key, string value) {
        //lock bucket first (unique), then ordered index (unique). This global ordering avoids deadlocks.
        auto b_lock = hash_.lock_bucket_unique(key);
        auto o_lock = ordered_.lock_unique();

        auto valptr = make_shared<string>(std::move(value));
        //use unlocked variants since we already hold the ordered-index lock.
        ordered_.put_unlocked(key, valptr);
        hash_put_unlocked(key, valptr);
    }

    //GET: read from hash index for O(1) average performance.
    bool get(const string &key, string &out) const {
        shared_ptr<string> val;
        if (!hash_.get(key, val)) return false;
        out = *val;
        return true;
    }

    //DELETE: remove from both indices under locks.
    bool erase(const string &key) {
        auto b_lock = hash_.lock_bucket_unique(key);
        auto o_lock = ordered_.lock_unique();
        bool ok1 = ordered_.erase_unlocked(key);
        bool ok2 = hash_erase_unlocked(key);
        return ok1 || ok2;
    }

    //ANGE: read from ordered index; returns up to `limit` results.
    vector<pair<string,string>> get_range(const string &lo, const string &hi, size_t limit = SIZE_MAX) const {
        return ordered_.range(lo, hi, limit);
    }

    size_t size() const { return ordered_.size(); }

private:
    //unlocked helpers (call only when holding the appropriate locks)
    void hash_put_unlocked(const string &key, const shared_ptr<string> &val) {
        hash_.unsafe_bucket_map(key)[key] = val;
    }
    bool hash_erase_unlocked(const string &key) {
        return hash_.unsafe_bucket_map(key).erase(key) > 0;
    }

    StripedHashIndex hash_;
    OrderedIndex ordered_;
};

//simple benchmark & REPL below

struct BenchResult {
    double seconds{0};
    size_t operations{0};
    double throughput_mops{0};
};

BenchResult micro_bench(KVStore &kv, size_t n_threads, size_t ops_per_thread, double put_ratio) {
    atomic<size_t> ready{0};
    vector<thread> ths;
    auto start = chrono::steady_clock::now();

    for (size_t t = 0; t < n_threads; ++t) {
        ths.emplace_back([&, t]{
            //simple LCG for reproducibility
            uint64_t x = 88172645463325252ull + t * 1315423911ull;
            auto rng = [&]{ x ^= x << 7; x ^= x >> 9; return x; };
            ready.fetch_add(1, memory_order_relaxed);
            while (ready.load(memory_order_acquire) < n_threads) {}

            for (size_t i = 0; i < ops_per_thread; ++i) {
                double r = (rng() % 1000) / 1000.0;
                string key = "k" + to_string((rng() % (ops_per_thread*2)));
                if (r < put_ratio) {
                    kv.put(key, "v" + to_string(rng()));
                }
                
                else if (r < put_ratio + 0.45) {
                    string out;
                    kv.get(key, out);
                }
                
                else if (r < put_ratio + 0.9) {
                    //range over a tiny window
                    auto lo = key;
                    auto hi = key + "~"; //tilde to include keys with same prefix
                    auto v = kv.get_range(lo, hi, 16);
                    (void)v;
                }
                
                else {
                    kv.erase(key);
                }
            }
        });
    }

    for (auto &th : ths) th.join();
    auto end = chrono::steady_clock::now();
    double secs = chrono::duration<double>(end - start).count();
    BenchResult br; br.seconds = secs; br.operations = n_threads * ops_per_thread; br.throughput_mops = br.operations / 1e6 / secs;
    return br;
}

void repl() {
    KVStore kv(128);
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cerr << "KVStore ready. Commands: PUT k v | GET k | DELETE k | RANGE lo hi [limit] | SIZE | BENCH n_threads ops_per_thread put_ratio | EXIT\n";

    string cmd;
    while (cin >> cmd) {
        if (cmd == "PUT") {
            string k; string v; if (!(cin >> k)) break; getline(cin, v); //remainder of line as value
            if (!v.empty() && v[0] == ' ') v.erase(0,1);
            kv.put(k, v);
            cout << "OK\n";
        } 
        
        else if (cmd == "GET") {
            string k; cin >> k; string out;
            if (kv.get(k, out)) cout << out << "\n"; else cout << "(nil)\n";
        } 
        
        else if (cmd == "DELETE") {
            string k; cin >> k; cout << (kv.erase(k) ? "(deleted)\n" : "(not found)\n");
        } 
        
        else if (cmd == "RANGE") {
            string lo, hi; cin >> lo >> hi; size_t limit = numeric_limits<size_t>::max();
            if (cin.peek() == ' ') cin >> limit;
            auto res = kv.get_range(lo, hi, limit);
            for (auto &p : res) cout << p.first << "=" << p.second << "\n";
            cout << "(" << res.size() << " results)\n";
        }
        
        else if (cmd == "SIZE") {
            cout << kv.size() << "\n";
        }
        
        else if (cmd == "BENCH") {
            size_t nt, ops; double pr; cin >> nt >> ops >> pr; auto br = micro_bench(kv, nt, ops, pr);
            cout << fixed << setprecision(3);
            cout << "time=" << br.seconds << "s ops=" << br.operations << " thrpt=" << br.throughput_mops << " Mops/s\n";
        }
        
        else if (cmd == "EXIT") {
            break;
        } 
        
        else {
            cout << "Unknown command\n";
        }
    }
}

int main() {
    repl();
    return 0;
}
