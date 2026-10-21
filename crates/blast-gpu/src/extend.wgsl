struct Job {
    qoff:u32, qlen:u32, soff:u32, slen:u32,
    scratch:u32, band:u32, xdrop:i32, max_cells:u32,
    reward:i32, penalty:i32, gap_open:i32, gap_extend:i32,
    limited:u32, pad0:u32, pad1:u32, pad2:u32,
}
struct Score { score:i32, q:u32, s:u32, edge:u32, status:u32, pad:u32 }
struct Params { active_jobs:u32, pad0:u32, pad1:u32, pad2:u32 }
@group(0) @binding(0) var<storage,read> dna:array<u32>;
@group(0) @binding(1) var<storage,read> jobs:array<Job>;
@group(0) @binding(2) var<storage,read_write> scratch:array<i32>;
@group(0) @binding(3) var<storage,read_write> output:array<Score>;
@group(0) @binding(4) var<uniform> params:Params;
const NEG:i32 = -1000000000;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=params.active_jobs {return;}
    let job=jobs[id.x];
    let width=job.slen+1u;
    var prev=job.scratch;
    var curr=prev+width;
    var prev_gap=curr+width;
    var curr_gap=prev_gap+width;
    // Only [prev_start, prev_end] is readable. Each following row writes its
    // entire evaluated interval, including dead internal cells. Never rely on
    // zero-initialized storage or on values left by a previous pooled request.
    scratch[prev]=0;
    scratch[prev_gap]=NEG;
    var prev_start=0u;
    let leading=u32(max(0,job.xdrop-job.gap_open)/job.gap_extend);
    var prev_end=min(min(job.slen,job.band),leading);
    for (var j=1u;j<=prev_end;j++) {
        let value= -job.gap_open-job.gap_extend*i32(j);
        scratch[prev+j]=value;
        scratch[prev_gap+j]=NEG;
    }
    var best=Score(0,0u,0u,0u,0u,0u);
    var visited=0u;
    let open=job.gap_open+job.gap_extend;
    for (var i=1u;i<=job.qlen;i++) {
        let band_start=u32(max(0,i32(i)-i32(job.band)));
        let end=min(job.slen,i+job.band);
        if band_start>end {break;}
        // Keep the original geometric budget, including the final dead row.
        // Skipping arithmetic must not change which jobs exceed max_cells.
        let row_cells=end-band_start+1u;
        if row_cells>job.max_cells-visited {best.status=1u;break;}
        visited+=row_cells;
        var horizontal=NEG;
        var left_h=NEG;
        // No local restart: the dead prefix cannot revive. This also holds for
        // column zero, whose gap cost only grows while best.score never falls.
        let start=max(band_start,prev_start);
        var first_live=job.slen+1u;
        var last_live=0u;
        for (var j=start;j<=end;j++) {
            var vertical=NEG;
            var h=NEG;
            if j==0u {
                h= -job.gap_open-job.gap_extend*i32(i);
                vertical=h;
            } else {
                var above=NEG;
                var above_gap=NEG;
                if j>=prev_start && j<=prev_end {above=scratch[prev+j];above_gap=scratch[prev_gap+j];}
                horizontal=max(left_h-open,horizontal-job.gap_extend);
                vertical=max(above-open,above_gap-job.gap_extend);
                var diag=NEG;
                if j-1u>=prev_start && j-1u<=prev_end {diag=scratch[prev+j-1u];}
                let a=dna[job.qoff+i-1u];
                let b=dna[job.soff+j-1u];
                let substitution=select(job.penalty,job.reward,a<4u && a==b);
                h=max(diag+substitution,max(vertical,horizontal));
            }
            if best.score-h>job.xdrop {h=NEG;vertical=NEG;horizontal=NEG;}
            else {
                first_live=min(first_live,j);
                last_live=j;
                if job.limited!=0u && abs(i32(i)-i32(j))==i32(job.band) && i<job.qlen && j<job.slen {best.edge=1u;}
                if h>best.score {best.score=h;best.q=i;best.s=j;}
            }
            scratch[curr+j]=h;
            scratch[curr_gap+j]=vertical;
            left_h=h;
            // prev_end + 1 can still receive a diagonal transition. Beyond it,
            // only the horizontal chain remains; once dead it cannot revive.
            if h==NEG && j>prev_end {break;}
        }
        let temp=prev;prev=curr;curr=temp;
        let temp_gap=prev_gap;prev_gap=curr_gap;curr_gap=temp_gap;
        if first_live>job.slen {break;}
        // Do not re-prune earlier cells against the final row best: retain the
        // original running-best X-drop order and strictly-first maximum ties.
        prev_start=first_live;prev_end=last_live;
    }
    output[id.x]=best;
}
