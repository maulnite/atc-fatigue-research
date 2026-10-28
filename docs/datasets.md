# Dataset aktif dan batas penggunaannya

Cakupan aktif tetap **seluruh eksperimen pada tiga notebook**, termasuk MRL, DROZY, dan audit CogBeacon. Ketujuh dataset lokal berikut masih diperlukan; tidak ada dataset utuh yang dapat dihapus sebagai “tidak dipakai” tanpa menghilangkan kemampuan mengulang bagian penelitian tersebut.

| Dataset | Dipakai untuk | Sumber lokal | Batas klaim |
|---|---|---|---|
| ATCOSIM | Domain audio ATC, QC, eGeMAPS/MFCC 114 D | `data/raw/atcosim/` | Tanpa fatigue labels; tidak menghasilkan fatigue accuracy |
| Predi-COVID | Supervised clinical fatigue/unwell dan clean/radio robustness | `data/raw/predi_covid/` | Populasi klinis COVID-19; bukan occupational ATC fatigue |
| MRL Eye | HOG/SVM eye-state classification | `data/raw/mrl_eye/` | Foto mata; tidak mendukung pengukuran blink temporal/FPS |
| DROZY | EAR/blink/closure, test-level KSS Ridge/CNN/ablation | `data/raw/drozy/` | Sleepiness KSS per test, bukan label fatigue independen per window |
| CogBeacon | Audit deskriptif self-report events dan landmarks | `data/raw/cogbeacon/` | Tidak dipaksakan menjadi classifier fatigue per frame |
| UTA-RLDD | Tiga pretrained visual CNN, engineering pilot dan persiapan full benchmark | `data/raw/uta_rldd/` | Tiga kelas drowsiness 0/5/10; bukan KSS numerik atau validated ATC fatigue |
| RAVDESS | Genuine paired audio-video emotion; unimodal, early, late dan dual-stream fusion | `data/raw/ravdess/` | Emotion recognition; tidak boleh dilaporkan sebagai fatigue performance |

Dataset mentah tidak pernah dimasukkan ke GitHub. Memindahkan notebook atau mengganti README tidak mengganti sumber data, label, split, cache atau model eksperimen.

## Akses dan publikasi

Ketentuan berikut mengikuti audit sumber yang telah dicatat pada laporan penelitian:

- [Predi-COVID](https://zenodo.org/records/5937844): CC BY 4.0; subset tetap 207 rekaman/120 peserta. Contoh publik wajib atribusi dan mengikuti manifest yang ditinjau.
- [RAVDESS](https://zenodo.org/records/1188976): CC BY-NC-SA 4.0; subset 128 genuine AV recordings/8 aktor. Contoh berizin tidak mengizinkan publikasi seluruh full feature cache/model/data.
- [UTA-RLDD](https://sites.google.com/view/utarldd/home): official research access/citation; izin publikasi wajah tidak mencakup seluruh peserta. Source video dan face crops tetap lokal. Empat ZIP Fold 1–2 tersedia; enam ZIP Fold 3–5 masih source-quota pada pemeriksaan terakhir 25 Oktober 2026.
- ATCOSIM dan DROZY mengikuti ketentuan pemilik; media mentah tidak dipublikasikan. MRL digunakan untuk riset eye state. CogBeacon tetap audit, dengan batas label dan sampling yang terdokumentasi.

[Manifest contoh publik](../reports/media_publication.json) menyimpan atribusi, perubahan dan checksum; [galeri](../assets/demo_media/README.md) menyediakan contoh kecil yang telah disetujui.

## Menyimpan dan membersihkan data

Raw folders, manifests, cache, models dan completed runs memiliki fungsi berbeda. Arsip yang digunakan untuk source checksums, ingestion, audit atau resume masih diperlukan walaupun video/gambarnya sudah diekstrak. Khusus UTA, menghapus empat ZIP lokal akan merusak kemampuan reuse/resume saat sumber resmi sedang dibatasi.

Cleanup struktur repository tidak menghapus active datasets, pretrained weights atau completed experiment runs. Notebook demo awal dan spesifikasi pengembangan terdahulu dipindahkan dari working tree aktif; riwayat lengkap tetap tersedia pada Git dan backup lokal.

Dataset kandidat seperti FatigueView/LFD dan IEEE ATC corpus tetap didokumentasikan sebagai kebutuhan riset sesuai status aksesnya; metadata/audit bukan bukti video berlabel sudah tersedia atau sudah ditrain. [Feasibility report](visual_dataset_feasibility.md) · [Audit sumber](../reports/dataset_audit.md).
