import cv2
import numpy as np
import sys

# imread : 画像ファイルを読み込んで、多次元配列(numpy.ndarray)にする。
# imreadについて : https://kuroro.blog/python/wqh9VIEmRXS4ZAA7C4wd/
# 第一引数 : 画像のファイルパス
# 戻り値 : 行 x 列 x 色の三次元配列(numpy.ndarray)が返される。
img = cv2.imread("sample.jpg")

# 画像ファイルが正常に読み込めなかった場合、プログラムを終了する。
if img is None:
    sys.exit("Could not read the image.")

# cvtColor : 画像の色空間(色)の変更を行う関数。
# cvtColorについて : https://kuroro.blog/python/7IFCPLA4DzV8nUTchKsb/
# 第一引数 : 多次元配列(numpy.ndarray)
# 第二引数 : 変更前の画像の色空間(色)と、変更後の画像の色空間(色)を示す定数を設定。
# cv2.COLOR_BGR2GRAY : BGR(Blue, Green, Red)形式の色空間(色)を持つ画像をグレースケール画像へ変更する。
# グレースケールとは? : https://www.shinkohsha.co.jp/blog/monochrome-shirokuro-grayscale/
# 戻り値 : 多次元配列(numpy.ndarray)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold関数 : しきい値を用いて画素の色を示す値を2値化するための関数。

# 第一引数 : 多次元配列(numpy.ndarray)
# 第二引数 : しきい値。float型。150とする。

# 第三引数 : しきい値を超えた画素に対して、色を示す値を指定。float型。255とする。

# 第四引数 : 2値化するための条件のタイプを指定する。
# cv2.THRESH_BINARY : (画素の色を示す値 <= 第二引数)の場合、画素に対して、0の値を与える。(画素の色を示す値 > 第二引数)の場合、画素に対して、第三引数の値を与える。

# 戻り値 #################
# _ : しきい値を返す。150を返す。
# img : 多次元配列(numpy.ndarray)を返す。
#########################
_, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# HoughLinesP関数 : ハフ変換を用いて、画像内の線分を検出するために利用する関数。

# 第一引数(必須) : 多次元配列(numpy.ndarray)
# 第二引数(必須) : xcosθ + ysinθ = ρのρの値。float型。1以上の値を指定する。
# 第三引数(必須) : xcosθ + ysinθ = ρのθの値。float型。
# radiansについて : https://note.nkmk.me/python-numpy-sin-con-tan/
# 第四引数(必須) : しきい値。int型。
# maxLineGap : float型。2つの点が1つ線分上にある場合に、点と点の間の間隔が指定する数より小さければ、同一の線とみなす。
# maxLineGapの説明例) 小さい線分が大きい線分と重なっています。小さい線分の長さがmaxLineGapよりも小さい場合に、小さい線分を大きい線分と同じ線とみなします。

# 戻り値
# lines : 検出された線分一覧。[(線分開始x1, 線分開始y1, 線分終了x1, 線分終了y2), ...]で返される。1つも線分が見つからない場合、Noneを返す。
lines = cv2.HoughLinesP(gray, 1, np.radians(1), 240, maxLineGap=50)

if lines is not None:
    # squeeze関数について : https://jellyware.jp/kurage/openvino/c06_numpy.html
    for x1, y1, x2, y2 in lines.squeeze():
        # line : 線分情報を元に、画像へ線分を書き込む関数
        # 第一引数 : 多次元配列(numpy.ndarray)
        # 第二引数 : 線分開始座標
        # 第三引数 : 線分終了座標
        #######################
        # 第四引数 : 線分の色情報。BGR(Blue, Green, Red)形式で指定。
        # ※ 検出される線分を消したい場合は、第四引数へ線分の周りと同系の色を指定ください。Chrome拡張機能の「ColorZilla」などを用いて、同系の色を調べると良いでしょう。(https://chrome.google.com/webstore/detail/colorzilla/bhlhnicpbhignbdhedgjhgdocnmhomnp/reviews?hl=ja)
        #######################
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0))

    # imwrite : 画像の保存を行う関数
    # 第一引数 : 保存先の画像ファイル名
    # 第二引数 : 多次元配列(numpy.ndarray)
    # <第二引数の例>
    # [
    # [
    # [234 237 228]
    # ...
    # [202 209 194]
    # ]
    # [
    # [10 27 16]
    # ...
    # [36 67 46]
    # ]
    # [
    # [34 51 40]
    # ...
    # [50 81 60]
    # ]
    # ]
    # imwriteについて : https://kuroro.blog/python/i0tNE1Mp8aEz8Z7n6Ggg/
    cv2.imwrite('output.png', img)
