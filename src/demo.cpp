#include "mnist.h"
#include <string>
#include <random>

extern "C" {
    void   layer_forward(const float* W, const float* x, const float* b,
                         float* out, size_t rows, size_t cols);
    void   matvec(const float* W, const float* x, float* out, size_t rows, size_t cols);
    void   vec_add(const float* a, const float* b, float* out, size_t n);
    void   softmax(const float* in, float* out, size_t n);
    size_t argmax(const float* x, size_t n);
}

static bool read_floats(FILE* f, std::vector<float>& v) {
    return std::fread(v.data(), sizeof(float), v.size(), f) == v.size();
}

static const char* HTML_HEAD = R"HTML(<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Assembly NN demo</title>
<style>
body{background:#111;color:#eee;font-family:system-ui,sans-serif;margin:24px}
h1{font-size:20px;font-weight:600}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:16px}
.card{background:#1c1c1c;border-radius:10px;padding:14px;border:2px solid #333}
.card.ok{border-color:#2e7d32}.card.bad{border-color:#c62828}
canvas{width:168px;height:168px;image-rendering:pixelated;background:#000;display:block;margin:0 auto 10px}
.title{text-align:center;margin-bottom:8px;font-size:14px}
.row{display:flex;align-items:center;gap:6px;font-size:12px;margin:2px 0}
.row span:first-child{width:10px}.row span:last-child{width:44px;text-align:right}
.track{flex:1;background:#2a2a2a;height:10px;border-radius:5px;overflow:hidden}
.fill{height:100%;background:#546e7a}.pick .fill{background:#4caf50}
.bad .pick .fill{background:#ef5350}
</style></head><body>
<h1>Digit classifier running on hand-written assembly</h1>
<div id="grid"></div>
<script>
const data=[
)HTML";

static const char* HTML_TAIL = R"HTML(];
const grid=document.getElementById('grid');
data.forEach(function(d){
  const card=document.createElement('div');
  card.className='card '+(d.pred===d.label?'ok':'bad');
  const c=document.createElement('canvas');c.width=28;c.height=28;
  const ctx=c.getContext('2d');const img=ctx.createImageData(28,28);
  for(let i=0;i<784;i++){const v=d.pixels[i];
    img.data[i*4]=v;img.data[i*4+1]=v;img.data[i*4+2]=v;img.data[i*4+3]=255;}
  ctx.putImageData(img,0,0);
  card.appendChild(c);
  const t=document.createElement('div');t.className='title';
  t.textContent='True: '+d.label+'   Predicted: '+d.pred;
  card.appendChild(t);
  d.probs.forEach(function(p,k){
    const r=document.createElement('div');
    r.className='row'+(k===d.pred?' pick':'');
    r.innerHTML='<span>'+k+'</span><div class="track"><div class="fill" style="width:'
      +(p*100).toFixed(1)+'%"></div></div><span>'+(p*100).toFixed(1)+'%</span>';
    card.appendChild(r);
  });
  grid.appendChild(card);
});
</script></body></html>
)HTML";

int main(int argc, char** argv) {
    const int IN = 784, HID = 128, OUT = 10;
    bool pause = (argc < 2);          // ./demo all  -> no pauses

    std::vector<float> W1(HID * IN), b1(HID), W2(OUT * HID), b2(OUT);
    FILE* f = std::fopen("weights.bin", "rb");
    if (!f) { std::printf("weights.bin not found. Run ./train_mlp first.\n"); return 1; }
    bool ok = read_floats(f, W1) && read_floats(f, b1) && read_floats(f, W2) && read_floats(f, b2);
    std::fclose(f);
    if (!ok) { std::printf("weights.bin has the wrong size. Retrain.\n"); return 1; }

    int n_test, n_tmp;
    std::vector<float> test_x = load_images("data/t10k-images-idx3-ubyte", n_test);
    std::vector<int>   test_y = load_labels("data/t10k-labels-idx1-ubyte", n_tmp);

    std::vector<int> byDigit[10];                    // test image indices per digit
    for (int i = 0; i < n_test; i++) byDigit[test_y[i]].push_back(i);

    std::mt19937 rng(std::random_device{}());
    FILE* html = std::fopen("demo.html", "w");
    std::fputs(HTML_HEAD, html);

    const char* shades = " .:-=+*#%@";
    int right = 0;

    for (int d = 0; d < 10; d++) {
        int idx = byDigit[d][rng() % byDigit[d].size()];
        const float* x = &test_x[(size_t)idx * IN];

        // forward pass: all assembly
        float h[HID], scores[OUT], probs[OUT];
        layer_forward(W1.data(), x, b1.data(), h, HID, IN);
        matvec(W2.data(), h, scores, OUT, HID);
        vec_add(scores, b2.data(), scores, OUT);
        softmax(scores, probs, OUT);
        int pred = (int)argmax(probs, OUT);
        if (pred == d) right++;

        // console output
        std::printf("\n=== Digit %d  (test image #%d) ===\n\n", d, idx);
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                char ch = shades[(int)(x[r * 28 + c] * 9.0f)];
                std::putchar(ch); std::putchar(ch);   // doubled so it looks square
            }
            std::putchar('\n');
        }
        std::printf("\nThe network's confidence for each digit:\n");
        for (int k = 0; k < OUT; k++) {
            std::string bar((int)(probs[k] * 30.0f + 0.5f), '#');
            std::printf("  %d  %-30s %5.1f%%%s\n", k, bar.c_str(), probs[k] * 100.0f,
                        k == pred ? "   <-- picks this" : "");
        }
        std::printf("\nPrediction: %d  %s\n", pred, pred == d ? "(correct)" : "(WRONG)");

        // same data, written for the web page
        std::fprintf(html, "{label:%d,pred:%d,probs:[", d, pred);
        for (int k = 0; k < OUT; k++) std::fprintf(html, "%s%.4f", k ? "," : "", probs[k]);
        std::fprintf(html, "],pixels:[");
        for (int i = 0; i < IN; i++) std::fprintf(html, "%s%d", i ? "," : "", (int)(x[i] * 255.0f + 0.5f));
        std::fprintf(html, "]},\n");

        if (pause && d < 9) { std::printf("\nPress Enter for the next digit..."); std::fflush(stdout); std::getchar(); }
    }

    std::printf("\n%d/10 correct\n", right);
    std::fputs(HTML_TAIL, html);
    std::fclose(html);
    std::printf("Wrote demo.html (open it in a browser)\n");
    return 0;
}