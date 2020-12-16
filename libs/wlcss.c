#include "wlcss.h"

int32_t nT;
int32_t nS;

int32_t penalty;
int32_t reward;
int32_t accepteddist;

int32_t *matching_costs;


void wlcss_init(int32_t p, int32_t r, int32_t e, int32_t nt, int32_t ns){
    penalty = p;
    reward = r;
    accepteddist = e;
    nT = nt+1;
    nS = ns+1;
}

int32_t max(int32_t a, int32_t b){
    if (a>=b)
        return a;
    else 
        return b;
}

int32_t* wlcss(int32_t *t, int32_t *s){

    int *matching_costs = (int32_t *)malloc(nT * nS * sizeof(int32_t));

    for(int i=0;i<nT;i++){
        for(int j=0;j<nS;j++){
            int offset = i * nS + j;
            matching_costs[offset]=0;
        }
    }

    for(int32_t i=1;i<nT;i++){
        for(int32_t j=1;j<nS;j++){
            int32_t offset = i * nS + j;
            int32_t distance = abs(s[j-1]-t[i-1]);
            if (distance <= accepteddist){
                matching_costs[offset] = matching_costs[offset-nS-1]+reward;
            } else{
                matching_costs[offset] = max(matching_costs[offset-nS-1]-penalty*distance,
                    max(matching_costs[offset-1]-penalty*distance,
                        matching_costs[offset-nS]-penalty*distance));
            }
        }
    }

    return matching_costs;
}

void free_mem(int32_t* a){
    free(a);
}

int32_t main(){
    wlcss_init(1, 8, 0, 4, 16);
    int32_t t[4]={11, 12, 9, 10};
    int32_t s[16]={13, 11, 12, 9, 10, 12, 11, 11, 10, 10, 12, 9, 9, 11, 12, 10};
    matching_costs = wlcss(t, s);
    for (int32_t i=0;i<nT;i++){
        for (int32_t j=0;j<nS;j++){
            int32_t offset = i * nS + j;
            printf("%d \t", matching_costs[offset]);
        }
        printf("\n");
    }
    printf("\n");
    free(matching_costs);
    return 0;
} 
