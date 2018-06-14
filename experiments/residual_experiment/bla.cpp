#include <iostream>

using namespace std;

#define ing long long

signed main() {
    
    int a1,a2,l1,l2,c,x,y;    
    cin >> a1 >> l1 >> a2 >> l2;
    cin >> c;
    cin >> x >> y;
    for(int i=0;i<=c;++i) {
        int how_much;
        int a1_n = a1 + i*x;
        int l1_n = l1 + (c-i)*y;
        if(l2 % a1) how_much = l2 / a1_n + 1;
        else how_much = l2 / a1_n;
        how_much--;
        if(how_much * a2 < l1_n) {
            cout << 1 << '\n';
            return 0;
        }
    }
    cout << 0 << '\n';
    
    return 0;
}