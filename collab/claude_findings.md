# CLAUDE FINDINGS — Phan tich bottleneck & lo trinh upgrade (goc nhin model + kien truc)

- Vai tro: Chuyen gia Computer Vision / Multi-Object Tracking (thanh vien group study 3 agents)
- Ngay: 2026-11-03
- Du lieu dau vao da doc: `RESULT-OVERVIEW.md`, `summary_all_pipeline.md`, `improvement_proposals.md`
- Pham vi: chi phan tich va de xuat. KHONG thay doi bat ky file source nao cua du an.

---

## 0. TL;DR — Tra loi cau hoi cua giang vien

Cau hoi: "Ket qua 88% (precision) co toi uu / nang cap tot hon duoc nua khong?"

Tra loi: **Co, nhung precision 0.888 KHONG phai cho de toi uu.** Precision dang gan tran va he thong co rat nhieu du dia trade-off precision lay recall (FN chiem ~90% ngan sach sai so). Dung huong toi uu la:

1. **Recall 0.424 → 0.55-0.65 trong 1 tuan** (chi doi config + pretrained model), **→ 0.70-0.80 trong 1-3 tuan** (finetune).
2. **MOTA 0.360 → 0.45-0.53 (quick-win), → 0.65-0.75 (research).** SOTA MOT17-test MOTA ~80 (ByteTrack 80.3, BoT-SORT ~80.5) dung detector finetune rieng.
3. **IDF1 ~0.52 → 0.58-0.65 (doi tracker), → 0.68-0.78 (them appearance memory).**
4. **Hai tang chua duoc do that**: cross-camera association (dang simulation 60% ngau nhien + noise) va scoring (binary, bimodal). Can bien chung thanh thanh phan DO DUOC.

---

## 1. Phan tich bottleneck (dinh luong)

### 1.1 Phan ra ngan sach sai so (error budget)

Tren tong 111,897 GT boxes cua 7 sequence:

| Thanh phan sai so | Uoc tinh | Ty trong |
|---|---|---|
| FN (missed detection) | ~64,400 boxes (R = 0.424) | **~90%** |
| FP | ~6,700 boxes (P = 0.888) | ~9.4% |
| ID switches | 450 (36+65+95+76+80+35+63) | ~0.6% |

Cach uoc tinh: MOTA = 1 - (FN + FP + IDsw)/GT. MOTA tb 0.360 → tong sai so ~0.64 x GT; tru FN (0.576 x GT) va IDsw (0.004 x GT) → FP ~0.060 x GT.

**Ket luan: sai so ~90% la miss detection tai tang DETECTOR, xuat hien truoc khi tracker va cac tang sau kip dong gop.** Bat ky cai thien nao khong cham den recall deu bi tran o MOTA ~0.45-0.50.

### 1.2 Ba nguyen nhan recall thap (xep theo muc dong gop)

**(1) conf = 0.35 mau thuan voi triet ly ByteTrack — nguyen nhan lon nhat, de sua nhat.**
ByteTrack duoc thiet ke de GIU detection conf thap (0.05-0.15) roi khu phuc qua second-pass association voi track dang active. Pipeline hien loc conf >= 0.35 ngay tai detector output → pool low-score (nguoi nho / bi che co score 0.10-0.35) khong bao gio den tracker, lam second-pass vo dung.
Kiem chung cheo tu so lieu hien co: ty le detections/GT tung sequence khop gan dung voi recall (MOT17-02: 7.2 det/frame vs ~31 GT/frame ≈ 0.23, recall do duoc 0.203; MOT17-04: 17.2/45.3 ≈ 0.38, recall 0.351). → Mat that xay ra tai detection, khong phai tai matching.

**(2) Suc chua model + do phan giai.**
YOLOv8n = 3.2M params, COCO mAP 37.3 (AP_small ~22) — la model nho nhat ho. YOLOv8m = 25.9M params, mAP 50.2; gap lon nhat nam o small objects, dung kieu nguoi xa/nho trong MOT17-02/13. imgsz = 960 downsample anh goc 1920x1080 → nguoi xa nho hon nua.

**(3) Dac thu canh MOT17-02 va MOT17-13.**
- MOT17-02: quang truong dong, GT ~31 box/frame, rat nhieu nguoi xa (bbox nho) + occlusion day → recall 0.203 (thap nhat).
- MOT17-13: quay tu xa/khong cao, anh sang yeu (avg conf thap nhat 0.624), nguoi thua nhung nho → recall 0.231.
Ca hai canh: box nho → it pixel → score tap trung o vung 0.10-0.35 → bi conf=0.35 cat.

### 1.3 Loi do luong (measurement) — "free win" khong doi model

MOT17 GT chua nhieu box khong thuoc lop pedestrian can danh gia (distractor, static person, occluder, reflection) va box co visibility rat thap. Protocol chinh thuc (TrackEval / py-motmetrics voi ignore rules) BO QUA cac box nay khi tinh P/R/MOTA. Neu matcher tu viet trong NB10 dang so voi TOAN BO GT thi:
- Recall va MOTA bi danh gia thap he thong (uoc tinh hoi lai ~3-8 diem recall "gratis" khi chuyen sang protocol chuan).
- So sanh truc tiep voi so SOTA cong bo se khong cong bang.

**Hanh dong:** chuyen sang `motmetrics` hoac `TrackEval` voi ignore rules + them HOTA/DetA/AssA (reviewer hien dai yeu cau). Day cung la De xuat #1 trong `improvement_proposals.md` — xac nhan uu tien cao nhat, lam truoc bat ky upgrade nao de co baseline cong bang.

### 1.4 Hieu ung domino xuong cac tang sau

Recall thap → tracklet ngan / fragment (chi 364/704 = 51.7% tracklet pass filter min_frames=10; MOT17-10 chi 39.3%) → 704 local IDs cho khoang ~350 nguoi that → global ID trung nguoi → association phai xu ly nhieu du lieu ao, scoring bien "so nguoi" thanh "so tracklet".

**Ket luan phan 1: Khong sua detection thi moi tang sau deu bi tran. Thu tu dung: (i) sua protocol do luong, (ii) mo conf + tang capacity detector, (iii) tracker, (iv) roi moi den Re-ID / association / scoring.**

---

## 2. De xuat upgrade chi tiet

Moi de xuat gom: Co che / Gain ky vong (co suy luan) / Effort / Rui ro / Cach ablation kiem chung.

### 2.A Detection

#### A1 [QUICK-WIN] Ha conf 0.35 → 0.10-0.15, tang imgsz 960 → 1280 cho sequence 1080p

- **Co che:** Mo lai low-confidence pool cho second-pass association cua ByteTrack (co san trong code, chi dang doi). Tang do phan giai giup small-object score cao hon.
- **Gain ky vong:** Recall +8-15 diem absolute (pho score 0.10-0.35 chinh la nguoi nho/bi che). Precision ha ve ~0.80-0.85 — chap nhan duoc vi FN >> FP gap 15 lan va he thong co human-in-the-loop. MOTA +6-10 diem.
- **Effort:** 0.5 ngay (sweep + rerun NB03, NB10).
- **Rui ro:** IDsw tang mot chut do detection tham nhieu; ByteTrack second-pass va match_thresh=0.8 kiem soat duoc. Luu y FP tang anh huong den tracklet filter (min_frames=10 van loc duoc).
- **Ablation kiem chung:** Sweep conf ∈ {0.05, 0.10, 0.15, 0.25, 0.35} x imgsz ∈ {960, 1280}. Bao cao P/R/F1/MOTA/IDF1/IDsw + ms/frame. Chon diem operating theo F1 hoac MOTA, KHONG chon theo precision.

#### A2 [QUICK-WIN] YOLOv8n → YOLOv8s/m pretrained (COCO), khong train

- **Co che:** Tang capacity 3.2M → 11.2M (s) / 25.9M (m) params; AP_small tang manh nhat trong cac nhom kich thuoc.
- **Gain ky vong:** Recall 0.424 → ~0.55-0.62 (m, ket hop A1); MOTA → ~0.45-0.52. Tham khao khoang cach COCO mAP 37.3 → 50.2 chuyen sang MOT17 person class.
- **Effort:** 1 ngay (doi 1 dong config + rerun pipeline + do lai). Chon s neu GPU yeu, m neu co GPU >= 8GB.
- **Rui ro:** Latency x ~2-2.5 (v8m @960 ~25-45ms T4-class GPU — van real-time; CPU thi khong). Luat luan phai bao cao speed/accuracy tradeoff.
- **Ablation kiem chung:** Bang n/s/m/l cung detector config: P/R/F1, MOTA, IDF1, ms/frame. Day la bang "detector scaling" review rat thich.

#### A3 [RESEARCH] Finetune YOLOv8m/l tren MOT17 train-halves + CrowdHuman

- **Co che:** Domain adaptation vung nguoc pedestrian occlusion (MOT17) + nguon crowd dan (CrowdHuman). Day la cach cac phuong phap SOTA (ByteTrack dung YOLOX-X finetune) dat MOTA ~80.
- **Gain ky vong:** Recall → 0.70-0.80; MOTA → 0.65-0.75 theo protocol train-halves (chia nua sequence train/validation).
- **Effort:** 3-5 ngay (convert GT MOT17 sang format YOLO, setup train halves, train, eval harness).
- **Rui ro:** Data leakage neu train tren chinh cac sequence danh gia — PHAI chia halves chuan (VD: {01,03,06,07,08,12,14} train / {02,04,05,09,10,11,13} val hoac nguoc lai; voi bo 7 sequence hien co thi chia 4/3 va cross-check 2 chieu). Overfit nua train → dung CrowdHuman lam aug.
- **Ablation kiem chung:** So sanh zero-shot vs finetuned tren CUNG halves protocol; bao cao ca 2 chieu chia de chung minh khong le chon.

#### A4 [RESEARCH, tuy chon] RT-DETR (R50/R101)

- **Co che:** DETR transformer end-to-end, multiscale deformable attention manh voi object nho/che, khong phu thuoc NMS tuning.
- **Gain ky vong:** ≈ v8l hoac +1-3 AP; loi the lon trong canh rat dong. Khong nen uu tien hon A3.
- **Effort:** 3-7 ngay (tich hop score vao ByteTrack can chinh).
- **Rui ro:** Tai tich hop, tai lieu MOT it hon YOLO; latency cao hon (~40-70ms R50).
- **Ablation kiem chung:** Chay song song nhu 1 cot trong bang detector comparison; chi lam neu con thoi gian.

### 2.B Tracking

#### B1 [QUICK-WIN] ByteTrack → OC-SORT hoac BoT-SORT (thu vien `boxmot`)

- **Co che:** OC-SORT: observation-centric re-update + Kalman consistency — khu phuc identity sau occlusion ma khong can appearance. BoT-SORT: camera motion compensation (GMC) + appearance ReID tich hop vao matching. Ca hai drop-in qua thu vien `boxmot` cua Mikel Brostrom.
- **Gain ky vong:** IDsw −30-50% (450 → ~250-320); IDF1 0.52 → 0.58-0.65; MOTA +1-3 diem. Trong 7 sequence hien deu la static camera nen phan lon gain den tu occlusion handling va appearance term.
- **Effort:** 0.5-1 ngay (cau hinh + rerun).
- **Rui ro:** Thap. Luu y: BoT-SORT dung appearance nhung ReID crop chat luong thap co the hai — chay bien khong appearance (BoT-SORT no-reid / OC-SORT) lam baseline.
- **Ablation kiem chung:** Vuong ByteTrack vs OC-SORT vs BoT-SORT (±appearance) tren cung detector: MOTA, IDF1, HOTA, IDsw, so tracklet valid (do fragment giam → ty le valid tracklet tang tu 51.7% len ~60-70% la bang chung truc tiep).

#### B2 [RESEARCH] Appearance gallery dai han (StrongSORT / hybrid)

- **Co che:** Luu embedding EMA theo track, re-associate sau mat bang long time — giam fragmentation (1 nguoi nhieu track ID).
- **Gain ky vong:** Fragment −20-40%; IDF1 +5-10 diem nua; giam so global ID ao cho tang cross-camera.
- **Effort:** 3-5 ngay.
- **Rui ro:** Embedding drift, memory/compute tang; can tuning cache size.
- **Ablation kiem chung:** Do "so local track tren 1 GT identity" (fragmentation ratio) truoc/sau + IDF1.

### 2.C Re-ID

#### C1 [QUICK-WIN] Quality-weighted embedding thay mean don gian

- **Co che:** Tracklet embedding hien la mean cua top-5 crops; dang mang quality score — dung weighted mean / trimmed mean de giam anh huong crop mo, bi che.
- **Gain ky vong:** AUC proxy +1-3 diem; tang match o nhom Medium (27% associations hien nay).
- **Effort:** 0.5 ngay.
- **Rui ro:** Thap.
- **Ablation kiem chung:** So mean vs quality-weighted vs median tren cung bo embedding: ROC AUC, match rate theo nhom do kho.

#### C2 [RESEARCH] OSNet-x1-0 → TransReID / SOLIDER (DINOv2 phu)

- **Co che:** Transformer part-attention xu ly occlusion + viewpoint; SOLIDER pretrain co giao thuc tren ~6M anh nguoi. Tham khao (approx): Market-1501 R1: OSNet ~94.5 vs TransReID/SOLIDER ~95-96 (mAP +4-6); MSMT17 (kho, gan canh that): OSNet mAP ~53 vs TransReID/SOLIDER ~67-75 → gap that su the hien o dataset kho, dung la kieu cross-camera thuc te. DINOv2 feature tot nhung khong chuyen door cho person ReID bang cac model chuyen nganh — de lam baseline so sanh thi duoc, khong nen lam lua chon chinh.
- **Gain ky vong:** Chinh xac association cross-camera (khi co du lieu that) +5-15% so voi OSNet, tap trung o nhom medium/hard.
- **Effort:** 3-7 ngay (torchreid ho tro TransReID; SOLIDER can tich hop repo rieng).
- **Rui ro:** Inference nang hon; crop nho (bbox height >= 40px) sau resize 256x128 van du.
- **Ablation kiem chung:** Giu nguyen toan bo association pipeline, chi doi embedding model → do tren (i) Market-1501/MSMT17 mAP/R1, (ii) track-pair proxy tren MOT17, (iii) association P/R theo temporal-split protocol (muc 2.D). Fix seed simulation de so sanh cong bang.

#### Loi phuong phap luan Re-ID can sua kem (khong ton effort lon)

ROC/AUC hien tai trong NB10 dung positive pairs = cac cap do simulation TU GHEP → tuc do do "do tach biet cua noise mo phong", khong phai kha nang Re-ID that. Phai thay bang:
- **Track-pair protocol** (De xuat #2 cu, dung track lam GT identity), va
- **Temporal-split protocol** (muc 2.D) cho cross-camera.

### 2.D Cross-camera association — kien truc thay simulation

#### D1 [RESEARCH, core] Gated association: temporal gate + spatial gate + Hungarian

- **Co che:** Thay matching ngau nhien bang quy trinh co rang buoc vat ly:
  1. **Temporal gate:** chi xet cap (entrance t_in, counter t_out) voi |t_out − t_in − mu_travel| <= 2σ (voi mu=180s, σ=60s → cua ±120s quanh 180s, theo dung prior dang dung de mo phong).
  2. **Spatial gate:** topology camera hop le (entrance → counter, cho phep re-entry co dinh danh).
  3. **Score** = cosine similarity cua quality-weighted embedding (+ phat neu khong mutual nearest-neighbor).
  4. **Hungarian tren ma tran DA GATE** (khong phai ma tran day).
- **Gain ky vong:** Bien tang association tu "khong the kiem chung" thanh "do duoc P/R". Gate temporal loai bo cuong khoang ~70-90% cap sai theo ly thuyet (cua 240s trong doi so t ngang ~600-1000s) → precision association tang manh khi co du lieu that.
- **Effort:** 3-5 ngay (logic gate lai phan matching trong NB06; ticket evidence van co the giu simulation cho business layer).
- **Rui ro:** Thap; phan lon la viec thiet ke evaluation.
- **Cach kiem chung khong can du lieu moi — TEMPORAL-SPLIT PROTOCOL:**
  - Chia moi sequence MOT17 tai moc 50% thoi gian: nua dau = "camera A", nua sau = "camera B".
  - Track cung GT identity vuot moc chia → positive pairs that; khac identity → negative.
  - Chay association → do P/R/F1 + mAP cua module association tren "2 camera" nay.
  - Co the inject noise kho (de co san easy/medium/hard) de do robustness.
  - Day la dong gong phuong phap luan defensible cho luan van: "cross-camera association duoc danh gia bang temporal camera-split protocol tren MOT17" — reviewer chap nhan vi khong dua du lieu moi.

### 2.E Scoring — binary → continuous

#### E1 [RESEARCH] Continuous features + calibrated probability (giu rule-based lam baseline)

- **Co che:** Thay flags nhi phan bang features lien tuc:
  - dwell time (log-normal), time-excess ratio = max(0, (t_travel − t_max)/t_max) thay vi flag time_exceeded (hien tai current_time=1000s >> max_travel=600s lam flag BAO HOA cho tat ca moi nguoi — da duoc chi ra trong ablation time_heavy),
  - so lan thay tai quay ve, so re-entry,
  - association margin = (sim tot nhat − sim ke tiep) — do dam sure cua viec khop,
  - crowd density tai entry, trajectory entropy.
  - Model: logistic regression (de giai thich) hoac GBDT (chinh xac) + isotonic/Platt calibration. Giu rule-based lam baseline de so sanh + bao dam explainability cho human review.
- **Gain ky vong:** Phan phoi score co gradient (thay bimodal 0.47/0.96) → tune duoc alert budget theo van hanh (alerts/gour); watchlist co y nghia xep hang thay vi 15 case bien gioi.
- **Effort:** 3-5 ngay.
- **Rui ro (quan trong):** Calibrate tren nhan mo phong KHONG transfer sang thuc te — phai ghi ro trong luan van la "architecture validated tren synthetic labels"; duong dai hoc tu confirmed_tp/confirmed_fp trong human_review.csv (feedback loop) khi co du lieu that.
- **Ablation kiem chung:** So rule vs logistic vs GBDT: phan phoi score, AUC tren synthetic labels, duong cong alert-volume vs threshold; kiem scoring stability bang Monte Carlo (500 runs) theo dung khung co san.

---

## 3. Hai goi trien khai

### 3.1 QUICK-WIN (< 1 tuan, chi config + pretrained, khong train)

| Thu tu | Viec | Effort | Ghi chu |
|---|---|---|---|
| 1 | Chuyen eval sang py-motmetrics/TrackEval + ignore rules + them HOTA, IDF1 chuan | 0.5-1 ngay | Lam TRUOC de co baseline cong bang; sua lai bang so luan van |
| 2 | conf 0.35 → 0.10-0.15 (sweep) | 0.5 ngay | Kem ablation conf x imgsz |
| 3 | YOLOv8n → YOLOv8m pretrained; imgsz 1280 cho sequence 1080p | 1 ngay | Bang scaling n/s/m/l |
| 4 | ByteTrack → OC-SORT / BoT-SORT qua `boxmot` | 0.5-1 ngay | Ablation ± appearance |
| 5 | Quality-weighted tracklet embedding | 0.5 ngay | Kem so mean/median |

**Muc tieu so lieu sau goi QUICK-WIN:**

| Metric | Hien tai | Muc tieu |
|---|---|---|
| Recall | 0.424 | 0.55-0.65 |
| Precision | 0.888 | 0.80-0.86 (chap nhan giam) |
| MOTA | 0.360 | 0.45-0.53 |
| IDF1 | ~0.52 | 0.58-0.65 |
| ID switches | 450 | 250-320 |
| Valid tracklet ratio | 51.7% | ~60-70% |

### 3.2 RESEARCH-UPGRADE (1-3 tuan)

| Tuan | Viec | Ket qua dinh kem |
|---|---|---|
| 1 | Finetune YOLOv8m/l tren MOT17 halves + CrowdHuman (A3) | MOTA 0.65-0.75 (halves protocol), bang zero-shot vs finetuned |
| 2 | TransReID/SOLIDER (C2) + gated association + temporal-split evaluation (D1) | Association P/R that + mAP ReID; thay the AUC circular |
| 3 | Continuous scoring + calibration + human-review feedback loop (E1); optional: StrongSORT (B2), RT-DETR (A4) | Phan phoi score lien tuc, alert-budget curve |

**Muc tieu sau goi RESEARCH:**

| Metric | Quick-win | Research |
|---|---|---|
| Recall | 0.55-0.65 | 0.70-0.80 |
| MOTA | 0.45-0.53 | 0.65-0.75 |
| IDF1 | 0.58-0.65 | 0.68-0.78 |
| Association | Simulation, khong do duoc | P/R do duoc (temporal-split) |
| Scoring | Bimodal binary | Continuous, calibrated |

---

## 4. Tran so voi SOTA va cach trinh bay trong luan van

So SOTA MOT17-test (approx): SORT 59.8; DeepSORT 75.4; OC-SORT ~77.5; StrongSORT ~77; ByteTrack 80.3 (IDF1 77.3); BoT-SORT ~80.5 (IDF1 ~80).

Diem mau chot: **cac so ~80 deu dung detector finetune rieng tren MOT17 (private detection setting).** So YOLOv8n COCO-pretrained conf=0.35 voi chung la khong cong bang theo ca 2 chieu (model capacity + finetune + conf protocol). Luan van nen chon MOT MOT trong so:

- **Phuong an A — "competitive numbers":** adopt train-halves protocol + finetune detector (A3) → co the claim MOTA 0.65-0.75 va so voi literature cung setting. Do la lua chon de phong ve voi giang vien.
- **Phuong an B — "cross-domain robustness study":** giu zero-shot COCO pretrained, dinh vi he thong la kha nang hoat dong khong can du lieu ve mien (dung cho kich ban camera thuc te khong nhan), va so sanh tuong dong trong nhom zero-shot. Van phai sua protocol do luong (muc 1.3).

Trong ca hai truong hop, phai (i) tach ro Nhom A (GT-verified) va Nhom B (simulation) nhu RESULT-OVERVIEW.md da lam — dung thuc hanh nay, va (ii) thay AUC ReID circular bang track-pair + temporal-split.

---

## 5. Tham khao

- ByteTrack: Zhang et al., ECCV 2022 (MOT17-test MOTA 80.3) — co so ly thuyet cho conf thap.
- OC-SORT: Cao et al., CVPR 2023 (observation-centric re-update).
- BoT-SOR: Aharon et al., arXiv 2022 (GMC + appearance).
- StrongSORT: Du et al., TMM 2023 (EMA appearance memory).
- TransReID: He et al., ICCV 2021 (ViT Re-ID).
- SOLIDER: Chen et al., AAAI 2024 (pretrain co giao thuc nguoi).
- OSNet: Zhou et al., ICCV 2019 (baseline hien tai).
- HOTA: Luiten et al., IJCV 2021 (metric nen them).
- TrackEval / py-motmetrics: cong cu danh gia protocol chuan MOT17.
- boxmot (Mikel Brostrom): bo tracker drop-in cho YOLO.

---

## 6. Tom tat 1 cau tra loi giang vien

"Precision 0.888 la ket qua tot nhung da gan tran va khong con du dia dang ke; dai dia dia that su cua he thong la recall 0.424 (90% sai so la miss), MOTA 0.360 va IDF1 0.52 so voi SOTA ~0.80, cong voi hai tang chua duoc kiem chung (cross-camera, scoring). Trong 1 tuan chi bang config + pretrained model co the dua Recall len 0.55-0.65 va MOTA len 0.45-0.53; trong 1-3 tuan voi finetune detector, tracker co appearance, Re-ID transformer, association co temporal gate va scoring continuous, MOTA 0.65-0.75 va IDF1 0.68-0.78 la muc tieu co can cu, kem theo protocol danh gia moi (temporal-split, HOTA) de moi con so deu do duoc."
