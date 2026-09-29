# プロジェクタ入出力特性の可視化手法 — 文献調査まとめ（2026-08-15）

目的: プロジェクタの入出力特性（トーン応答・色度・加法性・色域・モデル誤差）を
表すために文献でどのような図が使われているかを整理し、本リポジトリで作成済みの
図（`docs/color_grid_response_20260812.md` §2.3b/§7.2c 参照）と突き合わせて、
何が標準で何が空白かを確定する。

調査系統: ①ディスプレイ/プロジェクタ測色特性化の古典と規格
（Berns / Kwak & MacDonald / Seime & Hardeberg / Wyble & Rosen / Brainard /
IEC / ICDM / gamut rings）②プロカム測光補償論文
（Nayar / Grossberg / Grundhöfer & Iwai / ADICT / CompenNet系 / サーベイ2本 /
Fujii / Ashdown）。一次PDFの図キャプション確認に基づく。出典は末尾。

---

## 1. 特性化文献における図の体系（何を診断する図か）

| 図の型 | 典型的な軸・正規化 | 診断対象 | 代表文献 |
| --- | --- | --- | --- |
| ①トーン応答曲線 (TRC/EOTF) | 入力コード vs 輝度（最大1正規化）。実測点+モデルフィット重ね。CRTはlog-log（GOG直線化）、**LCDのS字は線形軸でないと見えない**（Seime & Hardeberg明記） | ガンマ推定、モデル選択、黒オフセット、DLP白セグメント介入点（Wyble & Zhang Fig.5の折れ点） | Berns 1996, Brainard 2002 Fig.4, Kwak & MacDonald 2000, Seime & Hardeberg Fig.3 |
| ②色度恒常性/カラートラッキング | (a) xy/u'v' 図上の R/G/B/グレーランプの軌跡（矢印=レベル減少方向、**黒補正前後の対**、拡大インセット） (b) レベル vs Δu'v' 折れ線（AAPM TG196） | 「色度不変」前提の成否、**色ずれの原因がフレアか本質か**の切り分け、グレー軸の色被り | Seime & Hardeberg Fig.4/5, Wyble & Rosen Fig.5, Day–Taplin–Berns |
| ③加法性検査 | グレーランプ実測 vs R+G+B単色和の重ね描き + **差分の第2軸**（Wyble & Rosen Fig.3）。数値要約（Σ単色−白: LCD 0.33% vs DLP 56%） | 行列モデル（加法混色仮定）の採用可否、白セグメント/クロストーク | Wyble & Rosen Fig.3, Wyble & Zhang Fig.6, Seime & Hardeberg, Kwak & MacDonald |
| ④原色SPD（駆動レベル別） | 波長 vs 分光放射輝度（レベル2点以上、形状比較） | スカラー倍性（色度恒常の分光的根拠）、スクリーン分光反射率 | Brainard Fig.5, Seime & Hardeberg Fig.1/2 |
| ⑤色域: 色度図三角形 | xy/u'v' + スペクトル軌跡 + 参照色域。複数輝度レベル版も | 原色飽和度・カバレッジ（ただし**輝度次元を捨てる**） | Kwak & MacDonald、Masaoka 2021（xy面積の歪み批判） |
| ⑥色域: CIELAB 3D立体 / gamut rings | L\*a\*b\* 凸包/alpha shape（ICC3D）。gamut rings = L\*10刻みスライスを面積保存の同心リング化（ICDM IDMS 5.32 / CIE 246 / IEC 62977系に採用） | 輝度込みの色域体積・被覆（u'v'で98%に見えて体積51%の事例あり） | Farup & Hardeberg ICC3D, Masaoka 2018-19 |
| ⑦モデル/LUT品質: ΔEヒストグラム+累積 | 独立RGB格子テストセット、mean/90–95%/max併記、CDFを第2軸 | 全域精度と最悪ケース（「95%がΔE<2」型の主張） | Wyble & Rosen Fig.7/9/10/11, Wyble & Zhang（5³テスト） |
| ⑧ΔEマップ（空間/格子） | 画面位置ごと・格子座標ごとの擬似カラー | 空間非均一、LUTが悪い局所 | Kwak & MacDonald（9点）、Thomas 2010（等高線化） |
| ⑨3D LUT格子可視化 | 格子点を対象色空間に描き変形・裏返りを見る | 非単調・クリッピング・補間破綻 | キャリブレーションツール慣習（LightSpace等） |
| ⑩時間安定性 | 点灯後時間 vs 輝度/色度 | ウォームアップ・ドリフト | Kwak & MacDonald, Seime & Hardeberg |

特性化系の推奨最小セット（調査エージェントの結論）: ①線形軸TRC（+暗部用log/残差小図）
②u'v'軌跡の黒補正前後対 ③差分第2軸つき加法性 ④色度図+gamut ringsの2枚組
⑤独立格子でのΔEヒスト+CDF、プロジェクタ固有として ⑧空間均一マップ・⑩時間系列。

## 2. プロカム補償論文の図法 — 診断図は驚くほど少ない

補償論文本流の図は「**表面/非補償/補償画像/補償後の写真4連 + 集計表・棒グラフ**」が
定型で、特性そのものの診断図は僅少:

- **Nayar 2003**: V行列の**生数値**を写真に引き出し線で重ねる（Fig.5、ヒートマップ化すらしない）。応答曲線のプロットは無し
- **Grossberg 2004**: 本調査で唯一の明示的応答曲線（Fig.4b、G1チャンネルのみの散布+回帰）
- **Grundhöfer & Iwai TIP2015**: 画像単位のΔE00/SSIM棒グラフが評価の主役。「DLPはチャンネル間依存が強いので6³必要」と**言葉とサンプル数で間接表現するのみ**で、依存性自体の色空間図示は無い。per-pixel誤差は反転差分画像どまり
- **ADICT 2010 が例外で3D色空間図の宝庫**: 歪んだガマット立体の3Dメッシュ（LCD/LCoS/DLP並置、Fig.1）、**deformed lattice**（測定点格子がBézier再パラメータ化で直方体に矯正される3コマ、Fig.3）。Ashdown 2006 の CIELUV多面体・到達可能輝度域の3D図（Bimber STAR Fig.7）も同系
- **CompenNet系**: PSNR/RMSE/SSIM表 + 拡大パッチ比較。ΔEも誤差マップも使わない

### クロストーク・非加法性の図示: 文献の定番4種と空白

定番: (1) 混合行列の数値提示 (2) 単チャンネル応答散布+回帰 (3) ADICT系のガマット
立体・変形格子 (4) 集計棒グラフ/表。

**文献的に空白（使われていないが有効）**とエージェントが特定したもの:
- (a) 入力格子→測定色空間の**変位ベクトル場**（ADICTの格子図は対応線を引かない）
- (b) **加法予測と実測の残差曲線**（レベル分解の加法比）
- (c) **per-pixel / 格子ごとのΔEヒートマップ**（全論文で不在）
- (d) 単チャンネルランプの**色度図軌跡**（procam文献には無い。特性化文献では定番）

## 3. 本リポジトリの図との対応表

| 文献の型 | 我々の図（2026-08-12〜13作成） | 判定 |
| --- | --- | --- |
| ①TRC | `ramps_native.png`（4駆動×3出力）、`channel_response_families.png`（**36背景のファミリー化は文献に前例なし**） | 充足+拡張 |
| ③加法性 | `additivity.png`（白vs原色和+ヒスト）、`line_additivity.png`（**レベル分解の比曲線=空白(b)を充足**） | 充足+空白埋め |
| ⑦ΔEヒスト/CDF | `lut3d_error_cdf.png`（セット別CDF） | 充足 |
| ⑧格子ΔEマップ | `residual_slices_*.png`、`lut3d_error_slices.png`（**往復誤差場+到達殻断面は空白(c)の充足+拡張**） | 空白埋め |
| ⑨LUT格子/ADICT系立体 | `gamut_solid_*.png`（外殻サーフェス+理想立方体+稜線強調、回転GIF） | ADICT Fig.1/3と同系+読みやすさ改善 |
| 空白(a) ベクトル場 | `io_map_raw/residual.png`、`io_map_slices_residual.png`（6次元符号化） | **文献に無い新規** |
| ⑩時間安定性 | `drift_sentinels.png`（番兵時系列。ウォームアップ系列は未取得） | 部分的 |
| 検証（要求vs達成） | `lut3d_validation.png`、`lut3d_tps_vs_linear_paired.png`（ペア散布） | Wyble系の独立テスト思想に合致 |
| ②色度恒常性/軌跡 | **無し** | 欠落 |
| ⑤⑥色域図（u'v'三角形 / gamut rings / Lab立体） | 部分的（c空間の殻立体はあるが測色空間でない） | 欠落（要注意点あり↓） |
| ④SPD | 無し（分光計測器が無い） | 対象外 |

**注意**: ②⑤⑥は XYZ/Lab を要する。我々の計測はカメラRGB（白正規化+線形sRGB仮定の
擬似Lab）なので、「カメラ基準の色度 r/(r+g+b)」でのトラッキング図は作れるが、
測色的な xy/u'v'/gamut rings を正式に描くには分光放射計かカメラの測色較正
（RPCC等）が必要。論文化の際はこの限界の明記が必須。

## 4. 推奨 — このプロジェクトの報告・論文用の図セット

1. **既作成で主張の核になる図**: `channel_response_families.png`（1D不可能性の一望）、
   `line_additivity.png`（加法比のレベル依存 0.4×〜2.5×）、`gamut_solid_*.png`
   （せん断された到達殻）、`lut3d_error_slices.png`（誤差の地図）、
   `lut3d_error_cdf.png`（精度の分布）。変位場（io_map系）は新規性のある表現として
   一望図に有効。
2. **追加する価値が高い図（未作成）**:
   - **カメラ基準色度のトラッキング軌跡**（黒補正前後の対、矢印つき）— 特性化文献の
     定番②の翻案。フレア起因かどうかの切り分けが1見開きでできる
   - **加法性のWyble式重ね描き**（グレー実測 vs R+G+B和 + 差分第2軸）—
     `line_additivity` の比表示より読者に通じる形式
   - **ウォームアップ時系列**（点灯直後から30分の輝度/バランス）— 日漂移・
     アイリス過渡の議論（§7.2c）を支える
   - 測色較正を入れる場合のみ: u'v'三角形 + gamut rings
3. **図の設計原則**（文献から）: 暗部は線形軸で潰れる→log軸か残差小図を併設 /
   ヒストグラムにはCDFを第2軸で / 色度図は拡大インセット / 黒補正前後を対で /
   独立テストセットで評価し mean・95%・max を併記。

## 5. 出典

特性化系: Berns (Displays 1996) / Kwak & MacDonald (Displays 2000) /
Seime & Hardeberg (CIC 2002, JSID 2003) / Wyble & Rosen (JIST 2006) /
Wyble & Zhang (CIC 2003) / Brainard, Pelli & Robson "Display Characterization" /
Day–Taplin–Berns (CR&A 2004) / AAPM TG196 (Med.Phys. 2016) /
Farup & Hardeberg ICC3D (CIC 2002) / Masaoka (Opt. Express 2021, gamut rings) /
IEC 61966-4, ICDM IDMS 5.32, CIE 246。

プロカム系: Nayar et al. (PROCAMS 2003) / Grossberg et al. (CVPR 2004) /
Grundhöfer (CVPRW 2013) / Grundhöfer & Iwai (TIP 2015) / Sajadi et al. ADICT
(ECCV 2010) / Ashdown et al. (PROCAMS 2006) / Fujii et al. (CVPR 2005) /
Huang & Ling CompenNet (CVPR 2019) / CompenNeSt++ (TPAMI 2021) /
Bimber et al. STAR (CGF 2008) / Grundhöfer & Iwai サーベイ (CGF 2018)。
（各URLは調査エージェントの報告に記載。PDF一式はセッションのスクラッチパッドに取得済み。）
