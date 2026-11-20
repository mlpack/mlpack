## `KernelPCA`: kernel principal components analysis

The `KernelPCA` class implements kernel principal components analysis, a
nonlinear data preprocessing technique.  Kernel PCA can be used for
nonlinear dimensionality reduction, or to project non-linearly-separable data
into a new space that may be linearly separable.  For example, kernel PCA with
the Gaussian kernel can project the two concentric circles below into a new
space where they are trivially separated by a classifier such as
[`LogisticRegression`](logistic_regression.md).

<!-- Concentric circles dataset, before and after kernel PCA.  Since all the
     points are here in the markdown... this is pretty big! -->
<div style="text-align: center">
<svg width="520" height="290" viewBox="-10 -10 510 280" fill="none" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
  <!-- Points before kernel PCA. -->

  <!-- Inner circle. -->
  <circle cx="114.835" cy=" 113" r="2" fill="#880000" />
  <circle cx="98.3356" cy=" 108.4" r="2" fill="#880000" />
  <circle cx="113.536" cy=" 94.8201" r="2" fill="#880000" />
  <circle cx="120.923" cy=" 108.531" r="2" fill="#880000" />
  <circle cx="99.5078" cy=" 122.132" r="2" fill="#880000" />
  <circle cx="77.8758" cy=" 96.6153" r="2" fill="#880000" />
  <circle cx="112.133" cy=" 114.692" r="2" fill="#880000" />
  <circle cx="104.431" cy=" 98.6511" r="2" fill="#880000" />
  <circle cx="110.503" cy=" 112.303" r="2" fill="#880000" />
  <circle cx="90.5226" cy=" 109.895" r="2" fill="#880000" />
  <circle cx="89.3868" cy=" 103.296" r="2" fill="#880000" />
  <circle cx="116.45" cy=" 104.012" r="2" fill="#880000" />
  <circle cx="110.701" cy=" 85.514" r="2" fill="#880000" />
  <circle cx="101.93" cy=" 87.9348" r="2" fill="#880000" />
  <circle cx="108.907" cy=" 104.619" r="2" fill="#880000" />
  <circle cx="90.1513" cy=" 103.015" r="2" fill="#880000" />
  <circle cx="94.7871" cy=" 105.047" r="2" fill="#880000" />
  <circle cx="99.2739" cy=" 119.474" r="2" fill="#880000" />
  <circle cx="113.047" cy=" 113.017" r="2" fill="#880000" />
  <circle cx="100.982" cy=" 83.4197" r="2" fill="#880000" />
  <circle cx="105.949" cy=" 105.883" r="2" fill="#880000" />
  <circle cx="101.647" cy=" 104.94" r="2" fill="#880000" />
  <circle cx="106.983" cy=" 89.1878" r="2" fill="#880000" />
  <circle cx="101.481" cy=" 81.0784" r="2" fill="#880000" />
  <circle cx="90.8306" cy=" 116.898" r="2" fill="#880000" />
  <circle cx="95.5801" cy=" 98.7392" r="2" fill="#880000" />
  <circle cx="84.2407" cy=" 96.0279" r="2" fill="#880000" />
  <circle cx="105.077" cy=" 82.091" r="2" fill="#880000" />
  <circle cx="90.8434" cy=" 81.676" r="2" fill="#880000" />
  <circle cx="95.4535" cy=" 101.74" r="2" fill="#880000" />
  <circle cx="82.4504" cy=" 108.427" r="2" fill="#880000" />
  <circle cx="95.5703" cy=" 111.203" r="2" fill="#880000" />
  <circle cx="107.013" cy=" 86.6197" r="2" fill="#880000" />
  <circle cx="115.409" cy=" 110.044" r="2" fill="#880000" />
  <circle cx="119.116" cy=" 102.396" r="2" fill="#880000" />
  <circle cx="99.237" cy=" 110.561" r="2" fill="#880000" />
  <circle cx="104.558" cy=" 98.5243" r="2" fill="#880000" />
  <circle cx="120.597" cy=" 104.301" r="2" fill="#880000" />
  <circle cx="82.1535" cy=" 112.706" r="2" fill="#880000" />
  <circle cx="77.8728" cy=" 104.062" r="2" fill="#880000" />
  <circle cx="107.818" cy=" 117.287" r="2" fill="#880000" />
  <circle cx="90.0032" cy=" 85.1911" r="2" fill="#880000" />
  <circle cx="84.0869" cy=" 94.2972" r="2" fill="#880000" />
  <circle cx="109.238" cy=" 88.4959" r="2" fill="#880000" />
  <circle cx="113.645" cy=" 103.681" r="2" fill="#880000" />
  <circle cx="106.204" cy=" 90.9674" r="2" fill="#880000" />
  <circle cx="93.9292" cy=" 90.7989" r="2" fill="#880000" />
  <circle cx="98.3945" cy=" 107.175" r="2" fill="#880000" />
  <circle cx="107.106" cy=" 90.8493" r="2" fill="#880000" />
  <circle cx="88.5123" cy=" 98.3249" r="2" fill="#880000" />
  <circle cx="94.5119" cy=" 111.201" r="2" fill="#880000" />
  <circle cx="87.9795" cy=" 110.29" r="2" fill="#880000" />
  <circle cx="108.958" cy=" 109.593" r="2" fill="#880000" />
  <circle cx="111.96" cy=" 92.3868" r="2" fill="#880000" />
  <circle cx="77.995" cy=" 95.2409" r="2" fill="#880000" />
  <circle cx="79.2265" cy=" 96.104" r="2" fill="#880000" />
  <circle cx="99.4924" cy=" 85.3144" r="2" fill="#880000" />
  <circle cx="99.7404" cy=" 87.0498" r="2" fill="#880000" />
  <circle cx="96.3402" cy=" 87.8575" r="2" fill="#880000" />
  <circle cx="87.6962" cy=" 109.476" r="2" fill="#880000" />
  <circle cx="100.205" cy=" 77.6409" r="2" fill="#880000" />
  <circle cx="104.773" cy=" 121.79" r="2" fill="#880000" />
  <circle cx="112.802" cy=" 88.9699" r="2" fill="#880000" />
  <circle cx="116.54" cy=" 88.5709" r="2" fill="#880000" />
  <circle cx="104.622" cy=" 115.887" r="2" fill="#880000" />
  <circle cx="93.0371" cy=" 99.9974" r="2" fill="#880000" />
  <circle cx="99.3056" cy=" 106.24" r="2" fill="#880000" />
  <circle cx="118.726" cy=" 96.1008" r="2" fill="#880000" />
  <circle cx="107.469" cy=" 114.094" r="2" fill="#880000" />
  <circle cx="100.86" cy=" 116.891" r="2" fill="#880000" />
  <circle cx="97.3508" cy=" 87.7635" r="2" fill="#880000" />
  <circle cx="106.702" cy=" 102.199" r="2" fill="#880000" />
  <circle cx="112.402" cy=" 86.3995" r="2" fill="#880000" />
  <circle cx="102.3" cy=" 108.85" r="2" fill="#880000" />
  <circle cx="99.4155" cy=" 80.5088" r="2" fill="#880000" />
  <circle cx="97.7638" cy=" 119.387" r="2" fill="#880000" />
  <circle cx="105.132" cy=" 98.2671" r="2" fill="#880000" />
  <circle cx="114.435" cy=" 100.195" r="2" fill="#880000" />
  <circle cx="93.0002" cy=" 108.219" r="2" fill="#880000" />
  <circle cx="101.558" cy=" 107.343" r="2" fill="#880000" />
  <circle cx="94.1713" cy=" 106.005" r="2" fill="#880000" />
  <circle cx="113.36" cy=" 98.0769" r="2" fill="#880000" />
  <circle cx="113.271" cy=" 102.237" r="2" fill="#880000" />
  <circle cx="95.4196" cy=" 115.745" r="2" fill="#880000" />
  <circle cx="87.5561" cy=" 100.587" r="2" fill="#880000" />
  <circle cx="98.2023" cy=" 94.7901" r="2" fill="#880000" />
  <circle cx="110.693" cy=" 94.7607" r="2" fill="#880000" />
  <circle cx="89.5231" cy=" 116.155" r="2" fill="#880000" />
  <circle cx="106.896" cy=" 99.6402" r="2" fill="#880000" />
  <circle cx="100.479" cy=" 88.5839" r="2" fill="#880000" />
  <circle cx="90.4139" cy=" 98.0757" r="2" fill="#880000" />
  <circle cx="86.4842" cy=" 86.2159" r="2" fill="#880000" />
  <circle cx="110.677" cy=" 95.5767" r="2" fill="#880000" />
  <circle cx="116.098" cy=" 89.2173" r="2" fill="#880000" />
  <circle cx="89.9974" cy=" 94.4424" r="2" fill="#880000" />
  <circle cx="102.595" cy=" 90.2756" r="2" fill="#880000" />
  <circle cx="92.8452" cy=" 111.363" r="2" fill="#880000" />
  <circle cx="93.4947" cy=" 100.237" r="2" fill="#880000" />
  <circle cx="112.624" cy=" 84.0197" r="2" fill="#880000" />
  <circle cx="116.276" cy=" 86.3868" r="2" fill="#880000" />
  <circle cx="98.0723" cy=" 93.1742" r="2" fill="#880000" />
  <circle cx="120.724" cy=" 106.752" r="2" fill="#880000" />
  <circle cx="99.4467" cy=" 108.892" r="2" fill="#880000" />
  <circle cx="101.203" cy=" 111.548" r="2" fill="#880000" />
  <circle cx="87.0189" cy=" 102.45" r="2" fill="#880000" />
  <circle cx="98.7822" cy=" 95.2739" r="2" fill="#880000" />
  <circle cx="95.2577" cy=" 106.543" r="2" fill="#880000" />
  <circle cx="91.7489" cy=" 109.478" r="2" fill="#880000" />
  <circle cx="107.128" cy=" 105.329" r="2" fill="#880000" />
  <circle cx="87.4901" cy=" 104.035" r="2" fill="#880000" />
  <circle cx="108.699" cy=" 113.741" r="2" fill="#880000" />
  <circle cx="88.7444" cy=" 89.8308" r="2" fill="#880000" />
  <circle cx="89.1257" cy=" 98.3934" r="2" fill="#880000" />
  <circle cx="84.2334" cy=" 92.63" r="2" fill="#880000" />
  <circle cx="105.944" cy=" 103.652" r="2" fill="#880000" />
  <circle cx="101.655" cy=" 94.9549" r="2" fill="#880000" />
  <circle cx="92.7425" cy=" 102.226" r="2" fill="#880000" />
  <circle cx="90.8788" cy=" 105.105" r="2" fill="#880000" />
  <circle cx="89.8894" cy=" 93.201" r="2" fill="#880000" />
  <circle cx="93.5294" cy=" 101.315" r="2" fill="#880000" />
  <circle cx="99.6814" cy=" 107.84" r="2" fill="#880000" />
  <circle cx="90.6515" cy=" 100.761" r="2" fill="#880000" />
  <circle cx="106.372" cy=" 100.235" r="2" fill="#880000" />
  <circle cx="108.017" cy=" 94.6013" r="2" fill="#880000" />
  <circle cx="98.3494" cy=" 108.268" r="2" fill="#880000" />
  <circle cx="94.6501" cy=" 93.1854" r="2" fill="#880000" />
  <circle cx="111.011" cy=" 104.367" r="2" fill="#880000" />
  <circle cx="103.959" cy=" 93.4675" r="2" fill="#880000" />
  <circle cx="115.357" cy=" 95.1113" r="2" fill="#880000" />
  <circle cx="115.098" cy=" 110.743" r="2" fill="#880000" />
  <circle cx="85.3554" cy=" 116.396" r="2" fill="#880000" />
  <circle cx="101.715" cy=" 115.044" r="2" fill="#880000" />
  <circle cx="89.5545" cy=" 101.11" r="2" fill="#880000" />
  <circle cx="107.143" cy=" 106.787" r="2" fill="#880000" />
  <circle cx="92.8368" cy=" 105.008" r="2" fill="#880000" />
  <circle cx="99.9433" cy=" 95.4263" r="2" fill="#880000" />
  <circle cx="100.768" cy=" 109.725" r="2" fill="#880000" />
  <circle cx="106.706" cy=" 109.391" r="2" fill="#880000" />
  <circle cx="113.91" cy=" 89.1615" r="2" fill="#880000" />
  <circle cx="105.551" cy=" 112.729" r="2" fill="#880000" />
  <circle cx="107.854" cy=" 84.8267" r="2" fill="#880000" />
  <circle cx="111.841" cy=" 99.7066" r="2" fill="#880000" />
  <circle cx="121.441" cy=" 98.0444" r="2" fill="#880000" />
  <circle cx="102.688" cy=" 84.5812" r="2" fill="#880000" />
  <circle cx="101.012" cy=" 86.1379" r="2" fill="#880000" />
  <circle cx="110.594" cy=" 100.332" r="2" fill="#880000" />
  <circle cx="99.7717" cy=" 84.6204" r="2" fill="#880000" />
  <circle cx="120.496" cy=" 91.8619" r="2" fill="#880000" />
  <circle cx="118.013" cy=" 102" r="2" fill="#880000" />
  <circle cx="109.276" cy=" 95.4533" r="2" fill="#880000" />
  <circle cx="95.8028" cy=" 113.635" r="2" fill="#880000" />
  <circle cx="84.4691" cy=" 114.92" r="2" fill="#880000" />
  <circle cx="91.4892" cy=" 95.1181" r="2" fill="#880000" />
  <circle cx="103.506" cy=" 108.029" r="2" fill="#880000" />
  <circle cx="92.2332" cy=" 97.7289" r="2" fill="#880000" />
  <circle cx="110.123" cy=" 80.9617" r="2" fill="#880000" />
  <circle cx="112.655" cy=" 110.175" r="2" fill="#880000" />
  <circle cx="102.129" cy=" 87.6944" r="2" fill="#880000" />
  <circle cx="113.734" cy=" 100.49" r="2" fill="#880000" />
  <circle cx="96.7238" cy=" 104.95" r="2" fill="#880000" />
  <circle cx="84.638" cy=" 96.4549" r="2" fill="#880000" />
  <circle cx="110.911" cy=" 104.393" r="2" fill="#880000" />
  <circle cx="119.842" cy=" 96.516" r="2" fill="#880000" />
  <circle cx="91.4628" cy=" 113.256" r="2" fill="#880000" />
  <circle cx="112.615" cy=" 105.857" r="2" fill="#880000" />
  <circle cx="90.1182" cy=" 105.006" r="2" fill="#880000" />
  <circle cx="90.9614" cy=" 103.987" r="2" fill="#880000" />
  <circle cx="111.919" cy=" 106.348" r="2" fill="#880000" />
  <circle cx="119.074" cy=" 106.823" r="2" fill="#880000" />
  <circle cx="115.101" cy=" 104.47" r="2" fill="#880000" />
  <circle cx="91.9007" cy=" 108.416" r="2" fill="#880000" />
  <circle cx="97.5046" cy=" 107.153" r="2" fill="#880000" />
  <circle cx="97.3988" cy=" 114.567" r="2" fill="#880000" />
  <circle cx="95.984" cy=" 94.3516" r="2" fill="#880000" />
  <circle cx="94.0427" cy=" 86.6448" r="2" fill="#880000" />
  <circle cx="106.258" cy=" 97.5002" r="2" fill="#880000" />
  <circle cx="91.0686" cy=" 99.3135" r="2" fill="#880000" />
  <circle cx="92.8982" cy=" 103.499" r="2" fill="#880000" />
  <circle cx="119" cy=" 92.1052" r="2" fill="#880000" />
  <circle cx="84.0154" cy=" 91.7556" r="2" fill="#880000" />
  <circle cx="102.656" cy=" 106.408" r="2" fill="#880000" />
  <circle cx="102.515" cy=" 103.966" r="2" fill="#880000" />
  <circle cx="104.339" cy=" 84.5198" r="2" fill="#880000" />
  <circle cx="97.4722" cy=" 86.6408" r="2" fill="#880000" />
  <circle cx="107.845" cy=" 95.3073" r="2" fill="#880000" />
  <circle cx="107.653" cy=" 100.788" r="2" fill="#880000" />
  <circle cx="89.9271" cy=" 85.3729" r="2" fill="#880000" />
  <circle cx="87.896" cy=" 112.502" r="2" fill="#880000" />
  <circle cx="113.174" cy=" 101.265" r="2" fill="#880000" />
  <circle cx="102.07" cy=" 118.013" r="2" fill="#880000" />
  <circle cx="95.0729" cy=" 96.639" r="2" fill="#880000" />
  <circle cx="103.858" cy=" 77.67" r="2" fill="#880000" />
  <circle cx="95.8142" cy=" 79.5268" r="2" fill="#880000" />
  <circle cx="91.8359" cy=" 101.183" r="2" fill="#880000" />
  <circle cx="93.7403" cy=" 109.479" r="2" fill="#880000" />
  <circle cx="91.4446" cy=" 106.06" r="2" fill="#880000" />
  <circle cx="114.645" cy=" 92.3355" r="2" fill="#880000" />
  <circle cx="97.1814" cy=" 95.3524" r="2" fill="#880000" />
  <circle cx="86.6209" cy=" 101.687" r="2" fill="#880000" />
  <circle cx="100.916" cy=" 87.6875" r="2" fill="#880000" />
  <circle cx="111.463" cy=" 85.2746" r="2" fill="#880000" />
  <circle cx="97.1326" cy=" 117.925" r="2" fill="#880000" />
  <circle cx="95.6186" cy=" 96.1128" r="2" fill="#880000" />
  <circle cx="89.4747" cy=" 90.5266" r="2" fill="#880000" />
  <circle cx="110.022" cy=" 97.0083" r="2" fill="#880000" />
  <circle cx="83.3984" cy=" 84.6285" r="2" fill="#880000" />
  <circle cx="98.2167" cy=" 82.3744" r="2" fill="#880000" />
  <circle cx="110.583" cy=" 100.294" r="2" fill="#880000" />
  <circle cx="83.0393" cy=" 99.8518" r="2" fill="#880000" />
  <circle cx="94.5705" cy=" 97.4978" r="2" fill="#880000" />
  <circle cx="104.398" cy=" 83.3333" r="2" fill="#880000" />
  <circle cx="102.4" cy=" 105.282" r="2" fill="#880000" />
  <circle cx="98.4259" cy=" 90.825" r="2" fill="#880000" />
  <circle cx="100.722" cy=" 93.1147" r="2" fill="#880000" />
  <circle cx="103.765" cy=" 105.582" r="2" fill="#880000" />
  <circle cx="96.5536" cy=" 117.967" r="2" fill="#880000" />
  <circle cx="117.608" cy=" 97.9802" r="2" fill="#880000" />
  <circle cx="106.652" cy=" 88.6274" r="2" fill="#880000" />
  <circle cx="105.36" cy=" 114.76" r="2" fill="#880000" />
  <circle cx="101.236" cy=" 79.0481" r="2" fill="#880000" />
  <circle cx="91.5911" cy=" 119.504" r="2" fill="#880000" />
  <circle cx="117.838" cy=" 93.7263" r="2" fill="#880000" />
  <circle cx="112.565" cy=" 101.891" r="2" fill="#880000" />
  <circle cx="99.4454" cy=" 91.2797" r="2" fill="#880000" />
  <circle cx="78.1983" cy=" 94.6761" r="2" fill="#880000" />
  <circle cx="92.1339" cy=" 94.6696" r="2" fill="#880000" />
  <circle cx="100.89" cy=" 95.4873" r="2" fill="#880000" />
  <circle cx="107.332" cy=" 79.0491" r="2" fill="#880000" />
  <circle cx="107.962" cy=" 97.1526" r="2" fill="#880000" />
  <circle cx="119.204" cy=" 100.291" r="2" fill="#880000" />
  <circle cx="101.022" cy=" 88.0904" r="2" fill="#880000" />
  <circle cx="114.033" cy=" 103.079" r="2" fill="#880000" />
  <circle cx="90.4365" cy=" 98.3633" r="2" fill="#880000" />
  <circle cx="95.1342" cy=" 82.3715" r="2" fill="#880000" />
  <circle cx="116.64" cy=" 90.5809" r="2" fill="#880000" />
  <circle cx="96.2336" cy=" 103.001" r="2" fill="#880000" />
  <circle cx="89.8597" cy=" 111.494" r="2" fill="#880000" />
  <circle cx="111.013" cy=" 118.301" r="2" fill="#880000" />
  <circle cx="88.4555" cy=" 119.167" r="2" fill="#880000" />
  <circle cx="87.1824" cy=" 117.647" r="2" fill="#880000" />
  <circle cx="98.1219" cy=" 104.304" r="2" fill="#880000" />
  <circle cx="95.0594" cy=" 100.872" r="2" fill="#880000" />
  <circle cx="91.4199" cy=" 86.0728" r="2" fill="#880000" />
  <circle cx="105.787" cy=" 96.9015" r="2" fill="#880000" />
  <circle cx="115.662" cy=" 115.805" r="2" fill="#880000" />
  <circle cx="96.5622" cy=" 83.7746" r="2" fill="#880000" />
  <circle cx="94.4035" cy=" 98.4976" r="2" fill="#880000" />
  <circle cx="111.952" cy=" 103.395" r="2" fill="#880000" />
  <circle cx="100.408" cy=" 105.79" r="2" fill="#880000" />
  <circle cx="108.523" cy=" 94.3581" r="2" fill="#880000" />
  <!-- Outer circle. -->
  <circle cx="26.4081" cy=" 53.923" r="2" fill="#000088" />
  <circle cx="44.304" cy=" 17.2688" r="2" fill="#000088" />
  <circle cx="23.4629" cy=" 139.194" r="2" fill="#000088" />
  <circle cx="17.5812" cy=" 95.1699" r="2" fill="#000088" />
  <circle cx="33.0731" cy=" 87.4276" r="2" fill="#000088" />
  <circle cx="18.2073" cy=" 63.0905" r="2" fill="#000088" />
  <circle cx="21.7794" cy=" 59.9928" r="2" fill="#000088" />
  <circle cx="111.167" cy=" 35.2405" r="2" fill="#000088" />
  <circle cx="160.63" cy=" 38.5242" r="2" fill="#000088" />
  <circle cx="154.636" cy=" 162.526" r="2" fill="#000088" />
  <circle cx="18.7431" cy=" 44.4761" r="2" fill="#000088" />
  <circle cx="73.4187" cy=" 192.508" r="2" fill="#000088" />
  <circle cx="61.6844" cy=" 157.269" r="2" fill="#000088" />
  <circle cx="162.003" cy=" 36.9431" r="2" fill="#000088" />
  <circle cx="81.3713" cy=" 12.3091" r="2" fill="#000088" />
  <circle cx="42.9846" cy=" 128.596" r="2" fill="#000088" />
  <circle cx="17.7929" cy=" 61.9639" r="2" fill="#000088" />
  <circle cx="58.2706" cy=" 173.739" r="2" fill="#000088" />
  <circle cx="112.315" cy=" 193.26" r="2" fill="#000088" />
  <circle cx="57.8321" cy=" 39.6521" r="2" fill="#000088" />
  <circle cx="158.564" cy=" 74.582" r="2" fill="#000088" />
  <circle cx="107.933" cy=" 30.6967" r="2" fill="#000088" />
  <circle cx="158.152" cy=" 41.7711" r="2" fill="#000088" />
  <circle cx="89.5112" cy=" 168.937" r="2" fill="#000088" />
  <circle cx="33.1798" cy=" 160.383" r="2" fill="#000088" />
  <circle cx="146.887" cy=" 25.1064" r="2" fill="#000088" />
  <circle cx="156.289" cy=" 51.3851" r="2" fill="#000088" />
  <circle cx="121.494" cy=" 13.9119" r="2" fill="#000088" />
  <circle cx="154.196" cy=" 138.35" r="2" fill="#000088" />
  <circle cx="56.1602" cy=" 31.6489" r="2" fill="#000088" />
  <circle cx="7.78091" cy=" 135.311" r="2" fill="#000088" />
  <circle cx="154.103" cy=" 170.688" r="2" fill="#000088" />
  <circle cx="111.008" cy=" 33.3368" r="2" fill="#000088" />
  <circle cx="177.545" cy=" 115.364" r="2" fill="#000088" />
  <circle cx="162.675" cy=" 125.202" r="2" fill="#000088" />
  <circle cx="143.844" cy=" 49.7756" r="2" fill="#000088" />
  <circle cx="78.5749" cy=" 39.4164" r="2" fill="#000088" />
  <circle cx="97.8175" cy=" 199.937" r="2" fill="#000088" />
  <circle cx="33.0751" cy=" 46.6512" r="2" fill="#000088" />
  <circle cx="9.30287" cy=" 85.5736" r="2" fill="#000088" />
  <circle cx="163.628" cy=" 121.946" r="2" fill="#000088" />
  <circle cx="176.87" cy=" 110.671" r="2" fill="#000088" />
  <circle cx="27.2695" cy=" 78.3044" r="2" fill="#000088" />
  <circle cx="69.1518" cy=" 187.696" r="2" fill="#000088" />
  <circle cx="77.978" cy=" 9.64907" r="2" fill="#000088" />
  <circle cx="150.104" cy=" 155.179" r="2" fill="#000088" />
  <circle cx="162.636" cy=" 34.7075" r="2" fill="#000088" />
  <circle cx="11.2855" cy=" 125.635" r="2" fill="#000088" />
  <circle cx="148.541" cy=" 16.9769" r="2" fill="#000088" />
  <circle cx="40.3624" cy=" 74.3407" r="2" fill="#000088" />
  <circle cx="39.6659" cy=" 159.25" r="2" fill="#000088" />
  <circle cx="138.513" cy=" 179.018" r="2" fill="#000088" />
  <circle cx="26.7528" cy=" 132.208" r="2" fill="#000088" />
  <circle cx="26.3355" cy=" 122.768" r="2" fill="#000088" />
  <circle cx="17.217" cy=" 122.296" r="2" fill="#000088" />
  <circle cx="107.391" cy=" 175.324" r="2" fill="#000088" />
  <circle cx="48.553" cy=" 50.8446" r="2" fill="#000088" />
  <circle cx="87.3806" cy=" 37.2859" r="2" fill="#000088" />
  <circle cx="86.9105" cy=" 22.1708" r="2" fill="#000088" />
  <circle cx="98.6905" cy=" 6.16988" r="2" fill="#000088" />
  <circle cx="112.453" cy=" 25.5088" r="2" fill="#000088" />
  <circle cx="21.776" cy=" 160.157" r="2" fill="#000088" />
  <circle cx="154.767" cy=" 47.5561" r="2" fill="#000088" />
  <circle cx="100.148" cy=" 14.3862" r="2" fill="#000088" />
  <circle cx="74.3808" cy=" 25.1246" r="2" fill="#000088" />
  <circle cx="92.7938" cy=" 18.1406" r="2" fill="#000088" />
  <circle cx="114.222" cy=" 168.279" r="2" fill="#000088" />
  <circle cx="10.5594" cy=" 128.685" r="2" fill="#000088" />
  <circle cx="46.4413" cy=" 56.6673" r="2" fill="#000088" />
  <circle cx="146.98" cy=" 177.539" r="2" fill="#000088" />
  <circle cx="162.886" cy=" 28.3434" r="2" fill="#000088" />
  <circle cx="36.5772" cy=" 131.207" r="2" fill="#000088" />
  <circle cx="181.945" cy=" 85.4433" r="2" fill="#000088" />
  <circle cx="123.065" cy=" 189.004" r="2" fill="#000088" />
  <circle cx="142.351" cy=" 176.807" r="2" fill="#000088" />
  <circle cx="128.307" cy=" 20.076" r="2" fill="#000088" />
  <circle cx="165.371" cy=" 65.6266" r="2" fill="#000088" />
  <circle cx="112.812" cy=" 174.863" r="2" fill="#000088" />
  <circle cx="66.8417" cy=" 162.648" r="2" fill="#000088" />
  <circle cx="30.5013" cy=" 150.358" r="2" fill="#000088" />
  <circle cx="28.4613" cy=" 129.137" r="2" fill="#000088" />
  <circle cx="132.221" cy=" 191.802" r="2" fill="#000088" />
  <circle cx="56.5403" cy=" 148.124" r="2" fill="#000088" />
  <circle cx="79.5774" cy=" 185.556" r="2" fill="#000088" />
  <circle cx="133.004" cy=" 162.879" r="2" fill="#000088" />
  <circle cx="153.885" cy=" 161.901" r="2" fill="#000088" />
  <circle cx="164.269" cy=" 37.8154" r="2" fill="#000088" />
  <circle cx="113.346" cy=" 29.4787" r="2" fill="#000088" />
  <circle cx="23.354" cy=" 153.653" r="2" fill="#000088" />
  <circle cx="157.112" cy=" 66.0488" r="2" fill="#000088" />
  <circle cx="141.917" cy=" 38.0396" r="2" fill="#000088" />
  <circle cx="123.087" cy=" 24.7998" r="2" fill="#000088" />
  <circle cx="151.243" cy=" 60.5417" r="2" fill="#000088" />
  <circle cx="139.266" cy=" 173.427" r="2" fill="#000088" />
  <circle cx="148.338" cy=" 49.7965" r="2" fill="#000088" />
  <circle cx="87.2062" cy=" 12.2577" r="2" fill="#000088" />
  <circle cx="31.0556" cy=" 72.4536" r="2" fill="#000088" />
  <circle cx="110.342" cy=" 7.47456" r="2" fill="#000088" />
  <circle cx="94.8129" cy=" 165.315" r="2" fill="#000088" />
  <circle cx="179.215" cy=" 132.064" r="2" fill="#000088" />
  <circle cx="130.136" cy=" 20.1684" r="2" fill="#000088" />
  <circle cx="154.627" cy=" 53.7535" r="2" fill="#000088" />
  <circle cx="134.107" cy=" 25.9776" r="2" fill="#000088" />
  <circle cx="32.8297" cy=" 83.9755" r="2" fill="#000088" />
  <circle cx="148.721" cy=" 168.136" r="2" fill="#000088" />
  <circle cx="104.215" cy=" 25.2105" r="2" fill="#000088" />
  <circle cx="49.3948" cy=" 30.7028" r="2" fill="#000088" />
  <circle cx="131.506" cy=" 25.9972" r="2" fill="#000088" />
  <circle cx="24.8011" cy=" 159.736" r="2" fill="#000088" />
  <circle cx="72.963" cy=" 162.891" r="2" fill="#000088" />
  <circle cx="162.526" cy=" 155.213" r="2" fill="#000088" />
  <circle cx="65.703" cy=" 21.3579" r="2" fill="#000088" />
  <circle cx="132.813" cy=" 26.067" r="2" fill="#000088" />
  <circle cx="65.7328" cy=" 12.3328" r="2" fill="#000088" />
  <circle cx="79.6761" cy=" 189.587" r="2" fill="#000088" />
  <circle cx="168.095" cy=" 125.599" r="2" fill="#000088" />
  <circle cx="54.6136" cy=" 159.161" r="2" fill="#000088" />
  <circle cx="76.9601" cy=" 183.604" r="2" fill="#000088" />
  <circle cx="53.79" cy=" 162.196" r="2" fill="#000088" />
  <circle cx="18.3492" cy=" 127.864" r="2" fill="#000088" />
  <circle cx="186.281" cy=" 139.212" r="2" fill="#000088" />
  <circle cx="158.714" cy=" 61.5201" r="2" fill="#000088" />
  <circle cx="176.582" cy=" 148.334" r="2" fill="#000088" />
  <circle cx="157.559" cy=" 138.866" r="2" fill="#000088" />
  <circle cx="153.69" cy=" 34.139" r="2" fill="#000088" />
  <circle cx="49.8404" cy=" 57.1971" r="2" fill="#000088" />
  <circle cx="157.458" cy=" 47.2938" r="2" fill="#000088" />
  <circle cx="15.9089" cy=" 68.8749" r="2" fill="#000088" />
  <circle cx="30.0663" cy=" 42.2379" r="2" fill="#000088" />
  <circle cx="33.975" cy=" 108.63" r="2" fill="#000088" />
  <circle cx="133.959" cy=" 19.6502" r="2" fill="#000088" />
  <circle cx="178.818" cy=" 65.877" r="2" fill="#000088" />
  <circle cx="51.4099" cy=" 186.078" r="2" fill="#000088" />
  <circle cx="35.3706" cy=" 70.2824" r="2" fill="#000088" />
  <circle cx="23.2816" cy=" 58.4025" r="2" fill="#000088" />
  <circle cx="32.8909" cy=" 65.4519" r="2" fill="#000088" />
  <circle cx="107.584" cy=" 8.19332" r="2" fill="#000088" />
  <circle cx="16.4785" cy=" 112.484" r="2" fill="#000088" />
  <circle cx="37.9412" cy=" 24.104" r="2" fill="#000088" />
  <circle cx="121.88" cy=" 32.7489" r="2" fill="#000088" />
  <circle cx="61.4842" cy=" 184.408" r="2" fill="#000088" />
  <circle cx="110.694" cy=" 162.819" r="2" fill="#000088" />
  <circle cx="35.9554" cy=" 52.5839" r="2" fill="#000088" />
  <circle cx="20.5939" cy=" 43.1564" r="2" fill="#000088" />
  <circle cx="149.191" cy=" 146.64" r="2" fill="#000088" />
  <circle cx="52.3173" cy=" 53.2847" r="2" fill="#000088" />
  <circle cx="153.553" cy=" 176.137" r="2" fill="#000088" />
  <circle cx="24.8343" cy=" 51.6465" r="2" fill="#000088" />
  <circle cx="58.7832" cy=" 155.08" r="2" fill="#000088" />
  <circle cx="49.5074" cy=" 21.1107" r="2" fill="#000088" />
  <circle cx="41.1998" cy=" 180.644" r="2" fill="#000088" />
  <circle cx="111.16" cy=" 180.769" r="2" fill="#000088" />
  <circle cx="90.1137" cy=" 174.407" r="2" fill="#000088" />
  <circle cx="72.9299" cy=" 19.5192" r="2" fill="#000088" />
  <circle cx="157.003" cy=" 32.4619" r="2" fill="#000088" />
  <circle cx="91.7352" cy=" 28.3666" r="2" fill="#000088" />
  <circle cx="131.979" cy=" 20.5595" r="2" fill="#000088" />
  <circle cx="82.6885" cy=" 10.3071" r="2" fill="#000088" />
  <circle cx="42.8515" cy=" 130.137" r="2" fill="#000088" />
  <circle cx="156.417" cy=" 172.495" r="2" fill="#000088" />
  <circle cx="112.171" cy=" 189.644" r="2" fill="#000088" />
  <circle cx="180.234" cy=" 114.284" r="2" fill="#000088" />
  <circle cx="190.125" cy=" 83.6072" r="2" fill="#000088" />
  <circle cx="5.59901" cy=" 68.5535" r="2" fill="#000088" />
  <circle cx="156.022" cy=" 24.1316" r="2" fill="#000088" />
  <circle cx="103.627" cy=" 7.8232" r="2" fill="#000088" />
  <circle cx="63.558" cy=" 9.45714" r="2" fill="#000088" />
  <circle cx="54.4988" cy=" 179.44" r="2" fill="#000088" />
  <circle cx="189.871" cy=" 106.262" r="2" fill="#000088" />
  <circle cx="189.881" cy=" 71.3567" r="2" fill="#000088" />
  <circle cx="108.383" cy=" 168.255" r="2" fill="#000088" />
  <circle cx="87.9424" cy=" 11.3041" r="2" fill="#000088" />
  <circle cx="151.004" cy=" 59.7142" r="2" fill="#000088" />
  <circle cx="165.714" cy=" 97.8928" r="2" fill="#000088" />
  <circle cx="64.8412" cy=" 17.8216" r="2" fill="#000088" />
  <circle cx="157.294" cy=" 146.309" r="2" fill="#000088" />
  <circle cx="53.257" cy=" 42.6523" r="2" fill="#000088" />
  <circle cx="92.1179" cy=" 31.229" r="2" fill="#000088" />
  <circle cx="26.465" cy=" 139.045" r="2" fill="#000088" />
  <circle cx="39.0069" cy=" 59.6937" r="2" fill="#000088" />
  <circle cx="17.3094" cy=" 143.43" r="2" fill="#000088" />
  <circle cx="189.909" cy=" 118.523" r="2" fill="#000088" />
  <circle cx="111.283" cy=" 175.143" r="2" fill="#000088" />
  <circle cx="173.637" cy=" 92.6976" r="2" fill="#000088" />
  <circle cx="57.7548" cy=" 36.2141" r="2" fill="#000088" />
  <circle cx="186.088" cy=" 118.295" r="2" fill="#000088" />
  <circle cx="129.482" cy=" 171.251" r="2" fill="#000088" />
  <circle cx="171.266" cy=" 31.0123" r="2" fill="#000088" />
  <circle cx="9.43483" cy=" 101.97" r="2" fill="#000088" />
  <circle cx="89.1447" cy=" 195.054" r="2" fill="#000088" />
  <circle cx="164.809" cy=" 56.322" r="2" fill="#000088" />
  <circle cx="147.965" cy=" 56.2646" r="2" fill="#000088" />
  <circle cx="60.8971" cy=" 30.1696" r="2" fill="#000088" />
  <circle cx="33.9248" cy=" 100.818" r="2" fill="#000088" />
  <circle cx="25.993" cy=" 97.9617" r="2" fill="#000088" />
  <circle cx="182.174" cy=" 101.823" r="2" fill="#000088" />
  <circle cx="55.4488" cy=" 50.4725" r="2" fill="#000088" />
  <circle cx="8.73964" cy=" 100.544" r="2" fill="#000088" />
  <circle cx="168.767" cy=" 83.9796" r="2" fill="#000088" />
  <circle cx="66.7025" cy=" 161.619" r="2" fill="#000088" />
  <circle cx="29.4643" cy=" 78.9201" r="2" fill="#000088" />
  <circle cx="8.19768" cy=" 61.2273" r="2" fill="#000088" />
  <circle cx="26.2257" cy=" 143.492" r="2" fill="#000088" />
  <circle cx="144.947" cy=" 180.233" r="2" fill="#000088" />
  <circle cx="51.6056" cy=" 159.917" r="2" fill="#000088" />
  <circle cx="149.909" cy=" 183.457" r="2" fill="#000088" />
  <circle cx="28.3723" cy=" 38.753" r="2" fill="#000088" />
  <circle cx="166.281" cy=" 66.0102" r="2" fill="#000088" />
  <circle cx="115.057" cy=" 28.4301" r="2" fill="#000088" />
  <circle cx="96.6103" cy=" 187.428" r="2" fill="#000088" />
  <circle cx="76.7763" cy=" 193.47" r="2" fill="#000088" />
  <circle cx="155.72" cy=" 59.9773" r="2" fill="#000088" />
  <circle cx="123.528" cy=" 179.928" r="2" fill="#000088" />
  <circle cx="160.725" cy=" 152.999" r="2" fill="#000088" />
  <circle cx="32.9581" cy=" 33.9042" r="2" fill="#000088" />
  <circle cx="60.2482" cy=" 167.234" r="2" fill="#000088" />
  <circle cx="6.35097" cy=" 74.7577" r="2" fill="#000088" />
  <circle cx="168.369" cy=" 168.835" r="2" fill="#000088" />
  <circle cx="170.414" cy=" 127.175" r="2" fill="#000088" />
  <circle cx="38.322" cy=" 34.7659" r="2" fill="#000088" />
  <circle cx="63.4042" cy=" 162.875" r="2" fill="#000088" />
  <circle cx="169.307" cy=" 65.9337" r="2" fill="#000088" />
  <circle cx="177.775" cy=" 113.513" r="2" fill="#000088" />
  <circle cx="103.904" cy=" 18.9927" r="2" fill="#000088" />
  <circle cx="59.6897" cy=" 15.2508" r="2" fill="#000088" />
  <circle cx="187.58" cy=" 114.899" r="2" fill="#000088" />
  <circle cx="13.2605" cy=" 135.137" r="2" fill="#000088" />
  <circle cx="150.07" cy=" 147.755" r="2" fill="#000088" />
  <circle cx="133.392" cy=" 32.4935" r="2" fill="#000088" />
  <circle cx="134.77" cy=" 12.2721" r="2" fill="#000088" />
  <circle cx="143.471" cy=" 40.4986" r="2" fill="#000088" />
  <circle cx="121.911" cy=" 28.8644" r="2" fill="#000088" />
  <circle cx="39.8652" cy=" 21.0584" r="2" fill="#000088" />
  <circle cx="167.054" cy=" 101.249" r="2" fill="#000088" />
  <circle cx="82.8497" cy=" 14.398" r="2" fill="#000088" />
  <circle cx="166.953" cy=" 93.4317" r="2" fill="#000088" />
  <circle cx="76.3753" cy=" 170.047" r="2" fill="#000088" />
  <circle cx="58.0809" cy=" 46.4506" r="2" fill="#000088" />
  <circle cx="96.895" cy=" 176.379" r="2" fill="#000088" />
  <circle cx="15.9" cy=" 72.6716" r="2" fill="#000088" />
  <circle cx="33.1772" cy=" 50.7907" r="2" fill="#000088" />
  <circle cx="161.919" cy=" 85.2435" r="2" fill="#000088" />
  <circle cx="183.401" cy=" 78.2197" r="2" fill="#000088" />
  <circle cx="50.951" cy=" 33.8636" r="2" fill="#000088" />
  <circle cx="18.4992" cy=" 42.5443" r="2" fill="#000088" />
  <circle cx="74.6725" cy=" 28.7071" r="2" fill="#000088" />
  <circle cx="24.0541" cy=" 49.7773" r="2" fill="#000088" />
  <circle cx="143.591" cy=" 154.053" r="2" fill="#000088" />
  <circle cx="21.8403" cy=" 133.943" r="2" fill="#000088" />
  <circle cx="114.445" cy=" 164.033" r="2" fill="#000088" />

  <!-- Line between the diagrams. -->
  <line x1="250" x2="250" y1="0" y2="200" stroke="black" stroke-dasharray="2" />

  <!-- Points after kernel PCA. -->

  <!-- Inner circle. -->
  <circle cx="343.58" cy=" 192.938" r="2" fill="#880000" />
  <circle cx="306.144" cy=" 139.634" r="2" fill="#880000" />
  <circle cx="326.738" cy=" 116.552" r="2" fill="#880000" />
  <circle cx="356.829" cy=" 182.421" r="2" fill="#880000" />
  <circle cx="352.345" cy=" 185.325" r="2" fill="#880000" />
  <circle cx="356.281" cy=" 34.7033" r="2" fill="#880000" />
  <circle cx="340.427" cy=" 194.842" r="2" fill="#880000" />
  <circle cx="301.407" cy=" 108.441" r="2" fill="#880000" />
  <circle cx="329.371" cy=" 186.823" r="2" fill="#880000" />
  <circle cx="320.273" cy=" 120.261" r="2" fill="#880000" />
  <circle cx="313.337" cy=" 83.7148" r="2" fill="#880000" />
  <circle cx="333.996" cy=" 164.775" r="2" fill="#880000" />
  <circle cx="341.221" cy=" 67.323" r="2" fill="#880000" />
  <circle cx="320.008" cy=" 46.6651" r="2" fill="#880000" />
  <circle cx="310.95" cy=" 153.267" r="2" fill="#880000" />
  <circle cx="311.172" cy=" 84.4783" r="2" fill="#880000" />
  <circle cx="303.886" cy=" 110.305" r="2" fill="#880000" />
  <circle cx="341.203" cy=" 180.741" r="2" fill="#880000" />
  <circle cx="338.172" cy=" 191.746" r="2" fill="#880000" />
  <circle cx="335.601" cy=" 28.6808" r="2" fill="#880000" />
  <circle cx="306.52" cy=" 151.435" r="2" fill="#880000" />
  <circle cx="300.721" cy=" 133.301" r="2" fill="#880000" />
  <circle cx="322.092" cy=" 69.2239" r="2" fill="#880000" />
  <circle cx="345.143" cy=" 24.7154" r="2" fill="#880000" />
  <circle cx="340.284" cy=" 148.584" r="2" fill="#880000" />
  <circle cx="300.901" cy=" 78.4021" r="2" fill="#880000" />
  <circle cx="331.503" cy=" 37.2077" r="2" fill="#880000" />
  <circle cx="343.776" cy=" 38.0392" r="2" fill="#880000" />
  <circle cx="350.619" cy=" 2.93716" r="2" fill="#880000" />
  <circle cx="300.635" cy=" 94.549" r="2" fill="#880000" />
  <circle cx="342.262" cy=" 90.5729" r="2" fill="#880000" />
  <circle cx="314.877" cy=" 143.117" r="2" fill="#880000" />
  <circle cx="330.007" cy=" 58.8631" r="2" fill="#880000" />
  <circle cx="338.615" cy=" 185.378" r="2" fill="#880000" />
  <circle cx="343.683" cy=" 161.119" r="2" fill="#880000" />
  <circle cx="310.875" cy=" 152.511" r="2" fill="#880000" />
  <circle cx="301.646" cy=" 108.174" r="2" fill="#880000" />
  <circle cx="350.823" cy=" 169.179" r="2" fill="#880000" />
  <circle cx="352.139" cy=" 107.647" r="2" fill="#880000" />
  <circle cx="355.747" cy=" 63.8557" r="2" fill="#880000" />
  <circle cx="339.561" cy=" 194.812" r="2" fill="#880000" />
  <circle cx="339.565" cy=" 8.14356" r="2" fill="#880000" />
  <circle cx="334.243" cy=" 29.9603" r="2" fill="#880000" />
  <circle cx="328.581" cy=" 73.9842" r="2" fill="#880000" />
  <circle cx="323.765" cy=" 159.152" r="2" fill="#880000" />
  <circle cx="316.047" cy=" 74.5852" r="2" fill="#880000" />
  <circle cx="315.588" cy=" 34.2504" r="2" fill="#880000" />
  <circle cx="303.8" cy=" 133.807" r="2" fill="#880000" />
  <circle cx="317.903" cy=" 77.1538" r="2" fill="#880000" />
  <circle cx="315.655" cy=" 55.8149" r="2" fill="#880000" />
  <circle cx="316.167" cy=" 139.527" r="2" fill="#880000" />
  <circle cx="327.677" cy=" 113.967" r="2" fill="#880000" />
  <circle cx="319.06" cy=" 174.94" r="2" fill="#880000" />
  <circle cx="326.043" cy=" 100.347" r="2" fill="#880000" />
  <circle cx="357.051" cy=" 29.972" r="2" fill="#880000" />
  <circle cx="351.021" cy=" 33.2331" r="2" fill="#880000" />
  <circle cx="328.334" cy=" 29.706" r="2" fill="#880000" />
  <circle cx="322.301" cy=" 36.2818" r="2" fill="#880000" />
  <circle cx="321.126" cy=" 29.2557" r="2" fill="#880000" />
  <circle cx="326.713" cy=" 109.403" r="2" fill="#880000" />
  <circle cx="359.486" cy=" 16.3701" r="2" fill="#880000" />
  <circle cx="353.215" cy=" 195.79" r="2" fill="#880000" />
  <circle cx="336.485" cy=" 87.624" r="2" fill="#880000" />
  <circle cx="349.472" cy=" 96.9562" r="2" fill="#880000" />
  <circle cx="329.989" cy=" 185.87" r="2" fill="#880000" />
  <circle cx="304.254" cy=" 77.0887" r="2" fill="#880000" />
  <circle cx="302.037" cy=" 132.158" r="2" fill="#880000" />
  <circle cx="344.253" cy=" 134.041" r="2" fill="#880000" />
  <circle cx="328.114" cy=" 186.919" r="2" fill="#880000" />
  <circle cx="331.15" cy=" 179.237" r="2" fill="#880000" />
  <circle cx="320.695" cy=" 31.7435" r="2" fill="#880000" />
  <circle cx="304.613" cy=" 134.966" r="2" fill="#880000" />
  <circle cx="342.733" cy=" 76.1315" r="2" fill="#880000" />
  <circle cx="307.577" cy=" 154.591" r="2" fill="#880000" />
  <circle cx="347.187" cy=" 18.0091" r="2" fill="#880000" />
  <circle cx="341.305" cy=" 176.453" r="2" fill="#880000" />
  <circle cx="302.602" cy=" 108.703" r="2" fill="#880000" />
  <circle cx="325.53" cy=" 144.641" r="2" fill="#880000" />
  <circle cx="311.663" cy=" 120.521" r="2" fill="#880000" />
  <circle cx="304.203" cy=" 145.17" r="2" fill="#880000" />
  <circle cx="306.009" cy=" 113.256" r="2" fill="#880000" />
  <circle cx="322.738" cy=" 132.031" r="2" fill="#880000" />
  <circle cx="321.775" cy=" 151.872" r="2" fill="#880000" />
  <circle cx="329.2" cy=" 159.798" r="2" fill="#880000" />
  <circle cx="317.775" cy=" 64.8692" r="2" fill="#880000" />
  <circle cx="303.097" cy=" 65.9571" r="2" fill="#880000" />
  <circle cx="318.284" cy=" 108.098" r="2" fill="#880000" />
  <circle cx="340.477" cy=" 141.989" r="2" fill="#880000" />
  <circle cx="304.885" cy=" 121.928" r="2" fill="#880000" />
  <circle cx="317.477" cy=" 44.4969" r="2" fill="#880000" />
  <circle cx="310.754" cy=" 59.3407" r="2" fill="#880000" />
  <circle cx="345.013" cy=" 6.72223" r="2" fill="#880000" />
  <circle cx="317.09" cy=" 112.178" r="2" fill="#880000" />
  <circle cx="346.354" cy=" 98.3444" r="2" fill="#880000" />
  <circle cx="315.94" cy=" 40.6466" r="2" fill="#880000" />
  <circle cx="313.578" cy=" 58.8473" r="2" fill="#880000" />
  <circle cx="319.179" cy=" 134.59" r="2" fill="#880000" />
  <circle cx="303.401" cy=" 79.8305" r="2" fill="#880000" />
  <circle cx="351.054" cy=" 68.5873" r="2" fill="#880000" />
  <circle cx="354.454" cy=" 87.9318" r="2" fill="#880000" />
  <circle cx="306.096" cy=" 57.3582" r="2" fill="#880000" />
  <circle cx="353.645" cy=" 177.284" r="2" fill="#880000" />
  <circle cx="306.896" cy=" 145.682" r="2" fill="#880000" />
  <circle cx="313.685" cy=" 162.752" r="2" fill="#880000" />
  <circle cx="319.821" cy=" 72.9647" r="2" fill="#880000" />
  <circle cx="302.137" cy=" 70.44" r="2" fill="#880000" />
  <circle cx="305.313" cy=" 119.777" r="2" fill="#880000" />
  <circle cx="316.673" cy=" 122.371" r="2" fill="#880000" />
  <circle cx="307.907" cy=" 152.027" r="2" fill="#880000" />
  <circle cx="319.343" cy=" 82.2168" r="2" fill="#880000" />
  <circle cx="329.39" cy=" 188.115" r="2" fill="#880000" />
  <circle cx="328.734" cy=" 19.6288" r="2" fill="#880000" />
  <circle cx="313.895" cy=" 57.6451" r="2" fill="#880000" />
  <circle cx="336.436" cy=" 23.8328" r="2" fill="#880000" />
  <circle cx="304.123" cy=" 140.253" r="2" fill="#880000" />
  <circle cx="302.952" cy=" 78.6864" r="2" fill="#880000" />
  <circle cx="305.031" cy=" 88.2602" r="2" fill="#880000" />
  <circle cx="311.22" cy=" 97.6958" r="2" fill="#880000" />
  <circle cx="318.355" cy=" 34.8859" r="2" fill="#880000" />
  <circle cx="303.353" cy=" 85.8287" r="2" fill="#880000" />
  <circle cx="304.73" cy=" 141.437" r="2" fill="#880000" />
  <circle cx="309.256" cy=" 74.0084" r="2" fill="#880000" />
  <circle cx="303.784" cy=" 123.475" r="2" fill="#880000" />
  <circle cx="312.021" cy=" 98.7721" r="2" fill="#880000" />
  <circle cx="305.871" cy=" 139.045" r="2" fill="#880000" />
  <circle cx="309.195" cy=" 46.8448" r="2" fill="#880000" />
  <circle cx="316.175" cy=" 157.028" r="2" fill="#880000" />
  <circle cx="307.43" cy=" 79.022" r="2" fill="#880000" />
  <circle cx="332.569" cy=" 122.564" r="2" fill="#880000" />
  <circle cx="339.035" cy=" 187.214" r="2" fill="#880000" />
  <circle cx="352.101" cy=" 129.818" r="2" fill="#880000" />
  <circle cx="324.791" cy=" 176.425" r="2" fill="#880000" />
  <circle cx="312.057" cy=" 72.7462" r="2" fill="#880000" />
  <circle cx="309.961" cy=" 158.934" r="2" fill="#880000" />
  <circle cx="306.997" cy=" 103.524" r="2" fill="#880000" />
  <circle cx="301.772" cy=" 75.2283" r="2" fill="#880000" />
  <circle cx="308.872" cy=" 153.744" r="2" fill="#880000" />
  <circle cx="314.09" cy=" 169.12" r="2" fill="#880000" />
  <circle cx="339.328" cy=" 91.8186" r="2" fill="#880000" />
  <circle cx="320.895" cy=" 178.746" r="2" fill="#880000" />
  <circle cx="337.598" cy=" 55.3386" r="2" fill="#880000" />
  <circle cx="317.158" cy=" 136.373" r="2" fill="#880000" />
  <circle cx="354.041" cy=" 145.976" r="2" fill="#880000" />
  <circle cx="332.016" cy=" 37.3153" r="2" fill="#880000" />
  <circle cx="325.595" cy=" 37.0493" r="2" fill="#880000" />
  <circle cx="313.419" cy=" 136.316" r="2" fill="#880000" />
  <circle cx="330.868" cy=" 28.438" r="2" fill="#880000" />
  <circle cx="357.303" cy=" 119.439" r="2" fill="#880000" />
  <circle cx="339.096" cy=" 158.378" r="2" fill="#880000" />
  <circle cx="313.631" cy=" 107.238" r="2" fill="#880000" />
  <circle cx="321.741" cy=" 153.71" r="2" fill="#880000" />
  <circle cx="350.294" cy=" 122.176" r="2" fill="#880000" />
  <circle cx="311.446" cy=" 47.4328" r="2" fill="#880000" />
  <circle cx="306.878" cy=" 154.482" r="2" fill="#880000" />
  <circle cx="306.94" cy=" 62.5909" r="2" fill="#880000" />
  <circle cx="356.341" cy=" 51.7669" r="2" fill="#880000" />
  <circle cx="330.062" cy=" 183.086" r="2" fill="#880000" />
  <circle cx="320.884" cy=" 46.384" r="2" fill="#880000" />
  <circle cx="323.086" cy=" 144.587" r="2" fill="#880000" />
  <circle cx="301.618" cy=" 116.464" r="2" fill="#880000" />
  <circle cx="329.645" cy=" 39.596" r="2" fill="#880000" />
  <circle cx="315.916" cy=" 156.927" r="2" fill="#880000" />
  <circle cx="348.411" cy=" 137.639" r="2" fill="#880000" />
  <circle cx="327.099" cy=" 137.815" r="2" fill="#880000" />
  <circle cx="322.53" cy=" 166.762" r="2" fill="#880000" />
  <circle cx="312.946" cy=" 94.8102" r="2" fill="#880000" />
  <circle cx="309.913" cy=" 92.0789" r="2" fill="#880000" />
  <circle cx="321.052" cy=" 167.587" r="2" fill="#880000" />
  <circle cx="347.011" cy=" 177.166" r="2" fill="#880000" />
  <circle cx="329.406" cy=" 164.88" r="2" fill="#880000" />
  <circle cx="314.144" cy=" 117.792" r="2" fill="#880000" />
  <circle cx="304.218" cy=" 130.644" r="2" fill="#880000" />
  <circle cx="323.51" cy=" 162.191" r="2" fill="#880000" />
  <circle cx="305.452" cy=" 56.5353" r="2" fill="#880000" />
  <circle cx="327.579" cy=" 19.4479" r="2" fill="#880000" />
  <circle cx="304.996" cy=" 108.293" r="2" fill="#880000" />
  <circle cx="308.535" cy=" 67.5581" r="2" fill="#880000" />
  <circle cx="305.464" cy=" 95.6563" r="2" fill="#880000" />
  <circle cx="350.943" cy=" 117.426" r="2" fill="#880000" />
  <circle cx="338.838" cy=" 20.5809" r="2" fill="#880000" />
  <circle cx="303.328" cy=" 144.085" r="2" fill="#880000" />
  <circle cx="300.268" cy=" 131.04" r="2" fill="#880000" />
  <circle cx="333.67" cy=" 42.5033" r="2" fill="#880000" />
  <circle cx="324.31" cy=" 28.1656" r="2" fill="#880000" />
  <circle cx="310.589" cy=" 101.868" r="2" fill="#880000" />
  <circle cx="306.221" cy=" 130.38" r="2" fill="#880000" />
  <circle cx="339.118" cy=" 8.45707" r="2" fill="#880000" />
  <circle cx="333.402" cy=" 123.214" r="2" fill="#880000" />
  <circle cx="321.234" cy=" 147.089" r="2" fill="#880000" />
  <circle cx="335.861" cy=" 184.894" r="2" fill="#880000" />
  <circle cx="303.273" cy=" 65.4582" r="2" fill="#880000" />
  <circle cx="361.004" cy=" 26.1865" r="2" fill="#880000" />
  <circle cx="352.842" cy=" 8.39672" r="2" fill="#880000" />
  <circle cx="306.591" cy=" 79.7813" r="2" fill="#880000" />
  <circle cx="313.067" cy=" 129.094" r="2" fill="#880000" />
  <circle cx="311.148" cy=" 104.459" r="2" fill="#880000" />
  <circle cx="334.59" cy=" 107.865" r="2" fill="#880000" />
  <circle cx="302.818" cy=" 65.4974" r="2" fill="#880000" />
  <circle cx="320.845" cy=" 68.1603" r="2" fill="#880000" />
  <circle cx="320.38" cy=" 42.3774" r="2" fill="#880000" />
  <circle cx="343.872" cy=" 69.0023" r="2" fill="#880000" />
  <circle cx="335.85" cy=" 171.318" r="2" fill="#880000" />
  <circle cx="303.234" cy=" 64.4196" r="2" fill="#880000" />
  <circle cx="325.153" cy=" 23.336" r="2" fill="#880000" />
  <circle cx="313.697" cy=" 117.577" r="2" fill="#880000" />
  <circle cx="359.056" cy=" 2.68767" r="2" fill="#880000" />
  <circle cx="339.792" cy=" 18.4523" r="2" fill="#880000" />
  <circle cx="313.394" cy=" 136.098" r="2" fill="#880000" />
  <circle cx="333.528" cy=" 52.3225" r="2" fill="#880000" />
  <circle cx="303.1" cy=" 68.431" r="2" fill="#880000" />
  <circle cx="338.192" cy=" 39.1524" r="2" fill="#880000" />
  <circle cx="301.581" cy=" 137.559" r="2" fill="#880000" />
  <circle cx="311.352" cy=" 47.4528" r="2" fill="#880000" />
  <circle cx="305.934" cy=" 65.9597" r="2" fill="#880000" />
  <circle cx="303.17" cy=" 143.441" r="2" fill="#880000" />
  <circle cx="336.407" cy=" 169.694" r="2" fill="#880000" />
  <circle cx="338.254" cy=" 140.409" r="2" fill="#880000" />
  <circle cx="323.139" cy=" 65.6727" r="2" fill="#880000" />
  <circle cx="327.01" cy=" 184.506" r="2" fill="#880000" />
  <circle cx="353.607" cy=" 20.6237" r="2" fill="#880000" />
  <circle cx="348.584" cy=" 158.141" r="2" fill="#880000" />
  <circle cx="343.76" cy=" 121.897" r="2" fill="#880000" />
  <circle cx="319.391" cy=" 148.762" r="2" fill="#880000" />
  <circle cx="309.979" cy=" 52.8162" r="2" fill="#880000" />
  <circle cx="356.825" cy=" 28.1008" r="2" fill="#880000" />
  <circle cx="310.783" cy=" 46.9569" r="2" fill="#880000" />
  <circle cx="301.842" cy=" 78.8427" r="2" fill="#880000" />
  <circle cx="359.093" cy=" 38.7297" r="2" fill="#880000" />
  <circle cx="308.633" cy=" 111.977" r="2" fill="#880000" />
  <circle cx="343.827" cy=" 152.684" r="2" fill="#880000" />
  <circle cx="319.134" cy=" 44.2925" r="2" fill="#880000" />
  <circle cx="324.704" cy=" 157.161" r="2" fill="#880000" />
  <circle cx="310.511" cy=" 60.8739" r="2" fill="#880000" />
  <circle cx="341.852" cy=" 11.2204" r="2" fill="#880000" />
  <circle cx="345.115" cy=" 105.455" r="2" fill="#880000" />
  <circle cx="300.342" cy=" 104.178" r="2" fill="#880000" />
  <circle cx="325.709" cy=" 125.24" r="2" fill="#880000" />
  <circle cx="349.642" cy=" 200" r="2" fill="#880000" />
  <circle cx="353.573" cy=" 147.488" r="2" fill="#880000" />
  <circle cx="351.229" cy=" 139.29" r="2" fill="#880000" />
  <circle cx="300" cy=" 117.85" r="2" fill="#880000" />
  <circle cx="300.997" cy=" 88.4212" r="2" fill="#880000" />
  <circle cx="333.753" cy=" 12.5323" r="2" fill="#880000" />
  <circle cx="304.786" cy=" 103.483" r="2" fill="#880000" />
  <circle cx="354.02" cy=" 198.553" r="2" fill="#880000" />
  <circle cx="335.202" cy=" 17.4918" r="2" fill="#880000" />
  <circle cx="302.597" cy=" 73.2729" r="2" fill="#880000" />
  <circle cx="318.178" cy=" 154.54" r="2" fill="#880000" />
  <circle cx="301.392" cy=" 133.572" r="2" fill="#880000" />
  <circle cx="313.524" cy=" 99.1904" r="2" fill="#880000" />
  <!-- Outer circle. -->
  <circle cx="499.465" cy=" 82.7578" r="2" fill="#000088" />
  <circle cx="492.288" cy=" 90.8615" r="2" fill="#000088" />
  <circle cx="492.544" cy=" 106.221" r="2" fill="#000088" />
  <circle cx="493.28" cy=" 92.9199" r="2" fill="#000088" />
  <circle cx="490.068" cy=" 82.605" r="2" fill="#000088" />
  <circle cx="497.37" cy=" 86.0045" r="2" fill="#000088" />
  <circle cx="498.493" cy=" 84.3343" r="2" fill="#000088" />
  <circle cx="489.95" cy=" 82.9392" r="2" fill="#000088" />
  <circle cx="496.502" cy=" 101.445" r="2" fill="#000088" />
  <circle cx="491.495" cy=" 121.02" r="2" fill="#000088" />
  <circle cx="495.331" cy=" 87.8061" r="2" fill="#000088" />
  <circle cx="489.279" cy=" 115.338" r="2" fill="#000088" />
  <circle cx="489.772" cy=" 120.429" r="2" fill="#000088" />
  <circle cx="495.813" cy=" 101.589" r="2" fill="#000088" />
  <circle cx="493.133" cy=" 91.5151" r="2" fill="#000088" />
  <circle cx="483.665" cy=" 104.171" r="2" fill="#000088" />
  <circle cx="497.232" cy=" 86.1879" r="2" fill="#000088" />
  <circle cx="493.256" cy=" 117.337" r="2" fill="#000088" />
  <circle cx="490.196" cy=" 117.968" r="2" fill="#000088" />
  <circle cx="494.972" cy=" 78.6682" r="2" fill="#000088" />
  <circle cx="487.136" cy=" 110.225" r="2" fill="#000088" />
  <circle cx="492.687" cy=" 85.0858" r="2" fill="#000088" />
  <circle cx="497.334" cy=" 101.209" r="2" fill="#000088" />
  <circle cx="490.137" cy=" 127.961" r="2" fill="#000088" />
  <circle cx="491.539" cy=" 110.704" r="2" fill="#000088" />
  <circle cx="497.292" cy=" 97.8868" r="2" fill="#000088" />
  <circle cx="496.156" cy=" 102.092" r="2" fill="#000088" />
  <circle cx="495.969" cy=" 94.1614" r="2" fill="#000088" />
  <circle cx="485.057" cy=" 129.225" r="2" fill="#000088" />
  <circle cx="496.747" cy=" 83.2379" r="2" fill="#000088" />
  <circle cx="488.456" cy=" 104.39" r="2" fill="#000088" />
  <circle cx="490.871" cy=" 118.81" r="2" fill="#000088" />
  <circle cx="491.541" cy=" 84.2527" r="2" fill="#000088" />
  <circle cx="491.47" cy=" 117.521" r="2" fill="#000088" />
  <circle cx="486.307" cy=" 125.402" r="2" fill="#000088" />
  <circle cx="492.672" cy=" 96.7976" r="2" fill="#000088" />
  <circle cx="485.87" cy=" 74.2916" r="2" fill="#000088" />
  <circle cx="487.647" cy=" 115.033" r="2" fill="#000088" />
  <circle cx="499.866" cy=" 81.6851" r="2" fill="#000088" />
  <circle cx="492.419" cy=" 93.1295" r="2" fill="#000088" />
  <circle cx="486.302" cy=" 124.918" r="2" fill="#000088" />
  <circle cx="491.545" cy=" 117.454" r="2" fill="#000088" />
  <circle cx="495.642" cy=" 83.526" r="2" fill="#000088" />
  <circle cx="490.732" cy=" 116.262" r="2" fill="#000088" />
  <circle cx="492.119" cy=" 92.4725" r="2" fill="#000088" />
  <circle cx="490.212" cy=" 125.166" r="2" fill="#000088" />
  <circle cx="495.113" cy=" 101.547" r="2" fill="#000088" />
  <circle cx="490.613" cy=" 102.842" r="2" fill="#000088" />
  <circle cx="493.717" cy=" 98.8768" r="2" fill="#000088" />
  <circle cx="489.284" cy=" 73.7932" r="2" fill="#000088" />
  <circle cx="492.798" cy=" 112.455" r="2" fill="#000088" />
  <circle cx="491.446" cy=" 119.967" r="2" fill="#000088" />
  <circle cx="492.607" cy=" 104.779" r="2" fill="#000088" />
  <circle cx="491.909" cy=" 101.568" r="2" fill="#000088" />
  <circle cx="492.132" cy=" 101.87" r="2" fill="#000088" />
  <circle cx="492.275" cy=" 126.421" r="2" fill="#000088" />
  <circle cx="495.347" cy=" 75.0108" r="2" fill="#000088" />
  <circle cx="485.613" cy=" 75.8685" r="2" fill="#000088" />
  <circle cx="494.146" cy=" 87.5345" r="2" fill="#000088" />
  <circle cx="491.899" cy=" 94.2944" r="2" fill="#000088" />
  <circle cx="495.912" cy=" 88.8514" r="2" fill="#000088" />
  <circle cx="488.388" cy=" 108.105" r="2" fill="#000088" />
  <circle cx="496.948" cy=" 101.019" r="2" fill="#000088" />
  <circle cx="494.654" cy=" 91.5613" r="2" fill="#000088" />
  <circle cx="494.379" cy=" 85.8681" r="2" fill="#000088" />
  <circle cx="494.547" cy=" 89.6281" r="2" fill="#000088" />
  <circle cx="489.196" cy=" 130.216" r="2" fill="#000088" />
  <circle cx="490.178" cy=" 103.436" r="2" fill="#000088" />
  <circle cx="493.981" cy=" 73.3149" r="2" fill="#000088" />
  <circle cx="490.67" cy=" 118.509" r="2" fill="#000088" />
  <circle cx="493.162" cy=" 101.311" r="2" fill="#000088" />
  <circle cx="489.502" cy=" 105.053" r="2" fill="#000088" />
  <circle cx="492.394" cy=" 111.998" r="2" fill="#000088" />
  <circle cx="490.737" cy=" 118.77" r="2" fill="#000088" />
  <circle cx="491.419" cy=" 119.897" r="2" fill="#000088" />
  <circle cx="497.805" cy=" 93.9815" r="2" fill="#000088" />
  <circle cx="493.972" cy=" 107.519" r="2" fill="#000088" />
  <circle cx="492.14" cy=" 126.369" r="2" fill="#000088" />
  <circle cx="491.187" cy=" 122.111" r="2" fill="#000088" />
  <circle cx="492.567" cy=" 109.279" r="2" fill="#000088" />
  <circle cx="492.098" cy=" 103.861" r="2" fill="#000088" />
  <circle cx="488.575" cy=" 116.19" r="2" fill="#000088" />
  <circle cx="485.847" cy=" 117.314" r="2" fill="#000088" />
  <circle cx="492.365" cy=" 119.036" r="2" fill="#000088" />
  <circle cx="489.128" cy=" 128.755" r="2" fill="#000088" />
  <circle cx="491.522" cy=" 121.462" r="2" fill="#000088" />
  <circle cx="495.172" cy=" 102.087" r="2" fill="#000088" />
  <circle cx="494.601" cy=" 87.2433" r="2" fill="#000088" />
  <circle cx="490.534" cy=" 108.152" r="2" fill="#000088" />
  <circle cx="490.439" cy=" 106.277" r="2" fill="#000088" />
  <circle cx="497.867" cy=" 95.8494" r="2" fill="#000088" />
  <circle cx="497.789" cy=" 91.62" r="2" fill="#000088" />
  <circle cx="489.715" cy=" 102.45" r="2" fill="#000088" />
  <circle cx="491.995" cy=" 121.905" r="2" fill="#000088" />
  <circle cx="494.707" cy=" 98.9798" r="2" fill="#000088" />
  <circle cx="493.369" cy=" 91.6824" r="2" fill="#000088" />
  <circle cx="496.193" cy=" 80.407" r="2" fill="#000088" />
  <circle cx="493.04" cy=" 94.6681" r="2" fill="#000088" />
  <circle cx="486.616" cy=" 131.646" r="2" fill="#000088" />
  <circle cx="490.343" cy=" 116.586" r="2" fill="#000088" />
  <circle cx="497.822" cy=" 94.3728" r="2" fill="#000088" />
  <circle cx="495.024" cy=" 102.034" r="2" fill="#000088" />
  <circle cx="498.677" cy=" 94.3723" r="2" fill="#000088" />
  <circle cx="491.361" cy=" 81.6296" r="2" fill="#000088" />
  <circle cx="491.741" cy=" 121.195" r="2" fill="#000088" />
  <circle cx="494.636" cy=" 87.4741" r="2" fill="#000088" />
  <circle cx="497.473" cy=" 84.2719" r="2" fill="#000088" />
  <circle cx="498.579" cy=" 93.6324" r="2" fill="#000088" />
  <circle cx="489.473" cy=" 108.712" r="2" fill="#000088" />
  <circle cx="489.643" cy=" 124.674" r="2" fill="#000088" />
  <circle cx="490.935" cy=" 119.952" r="2" fill="#000088" />
  <circle cx="494.969" cy=" 87.7092" r="2" fill="#000088" />
  <circle cx="498.644" cy=" 93.9914" r="2" fill="#000088" />
  <circle cx="492.451" cy=" 91.5023" r="2" fill="#000088" />
  <circle cx="491.067" cy=" 117.502" r="2" fill="#000088" />
  <circle cx="489.329" cy=" 121.845" r="2" fill="#000088" />
  <circle cx="492.446" cy=" 117.44" r="2" fill="#000088" />
  <circle cx="492.728" cy=" 119.265" r="2" fill="#000088" />
  <circle cx="493.179" cy=" 117.12" r="2" fill="#000088" />
  <circle cx="492.238" cy=" 103.319" r="2" fill="#000088" />
  <circle cx="487.542" cy=" 113.488" r="2" fill="#000088" />
  <circle cx="493.319" cy=" 105.157" r="2" fill="#000088" />
  <circle cx="488.936" cy=" 115.803" r="2" fill="#000088" />
  <circle cx="487.281" cy=" 126.456" r="2" fill="#000088" />
  <circle cx="497.676" cy=" 99.5375" r="2" fill="#000088" />
  <circle cx="491.012" cy=" 70.875" r="2" fill="#000088" />
  <circle cx="497.064" cy=" 101.741" r="2" fill="#000088" />
  <circle cx="496.342" cy=" 87.4374" r="2" fill="#000088" />
  <circle cx="498.767" cy=" 83.9281" r="2" fill="#000088" />
  <circle cx="486.459" cy=" 92.9557" r="2" fill="#000088" />
  <circle cx="497.523" cy=" 95.3374" r="2" fill="#000088" />
  <circle cx="492.937" cy=" 108.111" r="2" fill="#000088" />
  <circle cx="488.096" cy=" 112.693" r="2" fill="#000088" />
  <circle cx="494.991" cy=" 77.6204" r="2" fill="#000088" />
  <circle cx="498.876" cy=" 83.7115" r="2" fill="#000088" />
  <circle cx="497.693" cy=" 79.044" r="2" fill="#000088" />
  <circle cx="493.23" cy=" 94.2136" r="2" fill="#000088" />
  <circle cx="492.105" cy=" 99.009" r="2" fill="#000088" />
  <circle cx="494.244" cy=" 88.8688" r="2" fill="#000088" />
  <circle cx="495.129" cy=" 88.6311" r="2" fill="#000088" />
  <circle cx="490.848" cy=" 115.601" r="2" fill="#000088" />
  <circle cx="484.038" cy=" 135.32" r="2" fill="#000088" />
  <circle cx="499.65" cy=" 79.2546" r="2" fill="#000088" />
  <circle cx="495.815" cy=" 87.3211" r="2" fill="#000088" />
  <circle cx="486.341" cy=" 129.426" r="2" fill="#000088" />
  <circle cx="491.496" cy=" 71.7098" r="2" fill="#000088" />
  <circle cx="489.833" cy=" 117.28" r="2" fill="#000088" />
  <circle cx="498.971" cy=" 83.7484" r="2" fill="#000088" />
  <circle cx="489.551" cy=" 119.097" r="2" fill="#000088" />
  <circle cx="494.707" cy=" 88.5602" r="2" fill="#000088" />
  <circle cx="487.554" cy=" 111.26" r="2" fill="#000088" />
  <circle cx="492.942" cy=" 123.618" r="2" fill="#000088" />
  <circle cx="492.553" cy=" 125.368" r="2" fill="#000088" />
  <circle cx="494.467" cy=" 88.4895" r="2" fill="#000088" />
  <circle cx="496.52" cy=" 100.266" r="2" fill="#000088" />
  <circle cx="492.236" cy=" 84.1879" r="2" fill="#000088" />
  <circle cx="497.872" cy=" 94.7226" r="2" fill="#000088" />
  <circle cx="492.578" cy=" 92.2907" r="2" fill="#000088" />
  <circle cx="484.586" cy=" 105.074" r="2" fill="#000088" />
  <circle cx="490.091" cy=" 117.599" r="2" fill="#000088" />
  <circle cx="491.439" cy=" 119.59" r="2" fill="#000088" />
  <circle cx="491.589" cy=" 116.518" r="2" fill="#000088" />
  <circle cx="490.02" cy=" 110.287" r="2" fill="#000088" />
  <circle cx="491.533" cy=" 92.2414" r="2" fill="#000088" />
  <circle cx="494.321" cy=" 99.9959" r="2" fill="#000088" />
  <circle cx="492.865" cy=" 94.04" r="2" fill="#000088" />
  <circle cx="491.082" cy=" 92.7174" r="2" fill="#000088" />
  <circle cx="491.292" cy=" 115.085" r="2" fill="#000088" />
  <circle cx="490.245" cy=" 113.089" r="2" fill="#000088" />
  <circle cx="489.222" cy=" 108.387" r="2" fill="#000088" />
  <circle cx="488.802" cy=" 130.751" r="2" fill="#000088" />
  <circle cx="493.141" cy=" 92.0586" r="2" fill="#000088" />
  <circle cx="490.11" cy=" 102.106" r="2" fill="#000088" />
  <circle cx="486.682" cy=" 119.597" r="2" fill="#000088" />
  <circle cx="494.229" cy=" 89.2496" r="2" fill="#000088" />
  <circle cx="489.483" cy=" 124.457" r="2" fill="#000088" />
  <circle cx="495.955" cy=" 77.8814" r="2" fill="#000088" />
  <circle cx="490.531" cy=" 82.063" r="2" fill="#000088" />
  <circle cx="492.8" cy=" 106.544" r="2" fill="#000088" />
  <circle cx="497.097" cy=" 76.4463" r="2" fill="#000088" />
  <circle cx="490.704" cy=" 106.193" r="2" fill="#000088" />
  <circle cx="489.399" cy=" 113.495" r="2" fill="#000088" />
  <circle cx="492.212" cy=" 126.341" r="2" fill="#000088" />
  <circle cx="491.638" cy=" 115.305" r="2" fill="#000088" />
  <circle cx="495.917" cy=" 80.705" r="2" fill="#000088" />
  <circle cx="490.498" cy=" 114.698" r="2" fill="#000088" />
  <circle cx="491.714" cy=" 125.278" r="2" fill="#000088" />
  <circle cx="490.246" cy=" 102.668" r="2" fill="#000088" />
  <circle cx="491.034" cy=" 97.2282" r="2" fill="#000088" />
  <circle cx="489.508" cy=" 116.509" r="2" fill="#000088" />
  <circle cx="495.627" cy=" 105.039" r="2" fill="#000088" />
  <circle cx="490.466" cy=" 99.8347" r="2" fill="#000088" />
  <circle cx="495.924" cy=" 83.6644" r="2" fill="#000088" />
  <circle cx="486.569" cy=" 88.369" r="2" fill="#000088" />
  <circle cx="491.8" cy=" 90.9111" r="2" fill="#000088" />
  <circle cx="492.005" cy=" 114.778" r="2" fill="#000088" />
  <circle cx="490.899" cy=" 71.956" r="2" fill="#000088" />
  <circle cx="490.902" cy=" 97.0138" r="2" fill="#000088" />
  <circle cx="490.953" cy=" 113.658" r="2" fill="#000088" />
  <circle cx="490.684" cy=" 122.229" r="2" fill="#000088" />
  <circle cx="494.774" cy=" 82.4462" r="2" fill="#000088" />
  <circle cx="492.631" cy=" 90.8464" r="2" fill="#000088" />
  <circle cx="492.614" cy=" 107.412" r="2" fill="#000088" />
  <circle cx="490.328" cy=" 118.051" r="2" fill="#000088" />
  <circle cx="492.988" cy=" 116.347" r="2" fill="#000088" />
  <circle cx="488.428" cy=" 115.761" r="2" fill="#000088" />
  <circle cx="497.418" cy=" 85.602" r="2" fill="#000088" />
  <circle cx="494.048" cy=" 107.723" r="2" fill="#000088" />
  <circle cx="495.46" cy=" 88.2144" r="2" fill="#000088" />
  <circle cx="492.387" cy=" 120.481" r="2" fill="#000088" />
  <circle cx="489.231" cy=" 115.526" r="2" fill="#000088" />
  <circle cx="492.711" cy=" 103.894" r="2" fill="#000088" />
  <circle cx="492.455" cy=" 122.565" r="2" fill="#000088" />
  <circle cx="490.965" cy=" 121.157" r="2" fill="#000088" />
  <circle cx="497.064" cy=" 85.9343" r="2" fill="#000088" />
  <circle cx="493.491" cy=" 118.991" r="2" fill="#000088" />
  <circle cx="491.868" cy=" 92.4584" r="2" fill="#000088" />
  <circle cx="487.751" cy=" 114.888" r="2" fill="#000088" />
  <circle cx="490.182" cy=" 120.544" r="2" fill="#000088" />
  <circle cx="498.237" cy=" 84.2783" r="2" fill="#000088" />
  <circle cx="492.061" cy=" 120.683" r="2" fill="#000088" />
  <circle cx="494.336" cy=" 107.937" r="2" fill="#000088" />
  <circle cx="491.555" cy=" 117.339" r="2" fill="#000088" />
  <circle cx="495.497" cy=" 90.1758" r="2" fill="#000088" />
  <circle cx="493.193" cy=" 90.4768" r="2" fill="#000088" />
  <circle cx="490.411" cy=" 114.182" r="2" fill="#000088" />
  <circle cx="490.52" cy=" 104.633" r="2" fill="#000088" />
  <circle cx="487.351" cy=" 128.285" r="2" fill="#000088" />
  <circle cx="498.001" cy=" 93.1279" r="2" fill="#000088" />
  <circle cx="494.444" cy=" 96.8192" r="2" fill="#000088" />
  <circle cx="497.373" cy=" 96.415" r="2" fill="#000088" />
  <circle cx="496.808" cy=" 90.0609" r="2" fill="#000088" />
  <circle cx="493.243" cy=" 89.8584" r="2" fill="#000088" />
  <circle cx="487.517" cy=" 119.996" r="2" fill="#000088" />
  <circle cx="493.699" cy=" 90.744" r="2" fill="#000088" />
  <circle cx="488.225" cy=" 117.51" r="2" fill="#000088" />
  <circle cx="492.51" cy=" 123.669" r="2" fill="#000088" />
  <circle cx="491.623" cy=" 73.8455" r="2" fill="#000088" />
  <circle cx="492.757" cy=" 125.421" r="2" fill="#000088" />
  <circle cx="496.036" cy=" 87.8949" r="2" fill="#000088" />
  <circle cx="500" cy=" 80.6554" r="2" fill="#000088" />
  <circle cx="485.796" cy=" 115.436" r="2" fill="#000088" />
  <circle cx="492.043" cy=" 110.341" r="2" fill="#000088" />
  <circle cx="497.669" cy=" 82.669" r="2" fill="#000088" />
  <circle cx="494.676" cy=" 88.3808" r="2" fill="#000088" />
  <circle cx="493.599" cy=" 83.8685" r="2" fill="#000088" />
  <circle cx="498.53" cy=" 84.3955" r="2" fill="#000088" />
  <circle cx="487.776" cy=" 129.04" r="2" fill="#000088" />
  <circle cx="492.559" cy=" 104.927" r="2" fill="#000088" />
  <circle cx="485.849" cy=" 133.616" r="2" fill="#000088" />

  <!-- Labels for each dataset. -->
  <text x="100" y="240" text-anchor="middle" fill="black" font-style="italic">Before kernel PCA.</text>
  <text x="400" y="240" text-anchor="middle" fill="black" font-style="italic">After kernel PCA.</text>
</svg>
</div>

mlpack's `KernelPCA` class allows the use of arbitrary kernels and, because
kernel PCA scales quadratically with data size, accelerated approximate variants
such as the Nystroem method are implemented and supported.

#### Simple usage example:

```c++
// Use KernelPCA to map the two concentric circles dataset into a linearly
// separable space.
arma::mat data;
mlpack::Load("http://datasets.mlpack.org/circles.csv", data, mlpack::Fatal);

// Step 1: create object.
mlpack::KernelPCA<mlpack::GaussianKernel> kpca;

// Step 2: apply transform.
arma::mat transformedData;
kpca.Apply(data, transformedData, 2 /* new dimensionality */);

// Now any simple classifier or downstream technique could be applied.
std::cout << "Dimensionality of new data: " << transformedData.n_rows << "."
    << std::endl;
std::cout << "Bounding box of new data: [["
    << arma::min(transformedData.row(0)) << ", "
    << arma::min(transformedData.row(1)) << "], ["
    << arma::max(transformedData.row(0)) << ", "
    << arma::max(transformedData.row(1)) << "]]." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `KernelPCA` objects.
 * [`Apply()`](#applying-transformations): apply kernel PCA transformation to
   data.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage.
 * [Template parameters](#advanced-functionality-different-kernel-rules) for
   using different approximation strategies.
 * [Advanced examples](#advanced-examples) that use the Nystroem method to
   efficiently approximate the kernel matrix on larger datasets.

#### See also:

 * [`PCA`](pca.md): principal components analysis
 * [mlpack kernels](../core/kernels.md): list of kernels implemented by mlpack
 * [`NystroemMethod`](nystroem_method.md): sampling-based approximation strategy
   for kernel matrices
 * [`RADICAL`](radical.md): independent components analysis
 * [mlpack preprocessing utilities](../preprocessing.md)
 * [mlpack transformations](../transformations.md)
 * [Kernel principal component analysis on Wikipedia](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis)
 * [Kernel tricks and nonlinear dimensionality reduction via RBF kernel PCA](https://sebastianraschka.com/Articles/2014_kernel_pca.html)

### Constructors

 * `k = KernelPCA<KernelType>()`
   - Construct a `KernelPCA` object using the given `KernelType`.
   - `KernelType` must be either an [mlpack-supplied kernel](../core/kernels.md)
     or a [custom kernel class](../core/kernels.md#implement-a-custom-kernel).

 * `k = KernelPCA<KernelType>(kernel=KernelType(), centerTransformedData=false)`
   - Construct a `KernelPCA` object using the given `KernelType`, optionally
     passing an instantiated `KernelType` object and whether or not to center
     the transformed data around the origin.
   - `KernelType` must be either an [mlpack-supplied kernel](../core/kernels.md)
     or a [custom kernel class](../core/kernels.md#implement-a-custom-kernel).

 * `k = KernelPCA<KernelType, KernelRule>(kernel, centerTransformedData, rule=KernelRule())`
   - Construct a `KernelPCA` object using the given `KernelType` and
     [`KernelRule`](#advanced-functionality-different-kernel-rules) strategy for
     kernel matrix computation, optionally
     passing instantiated `KernelType` and `KernelRule` objects, and whether or
     not to center the transformed data around the origin.
   - `KernelType` must be either an [mlpack-supplied kernel](../core/kernels.md)
     or a [custom kernel class](../core/kernels.md#implement-a-custom-kernel).
   - `KernelRule` must be one mlpack's pre-supplied kernel rules, or a custom
     kernel rule that matches the API described in the
     [advanced functionality section](#advanced-functionality-different-kernel-rules).

#### Constructor parameters

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `KernelType` | [kernel class](../core/kernels.md) | Template parameter specifying which kernel to use for kernel PCA. | _(N/A)_ |
| `KernelRule` | [kernel rule class](#advanced-functionality-different-kernel-rules) | Template parameter specifying the strategy to use for computing the kernel matrix. | [`NaiveKernelRule`](#advanced-functionality-different-kernel-rules) |
|----------|----------|-----------------|-------------|
| `kernel` | [`KernelType`](../core/kernels.md) | Instantiated kernel object to use when applying kernel PCA. | `KernelType()` |
| `centerTransformedData` | `bool` | If `true`, then transformed data will be centered around the origin such that `mean(transformedData)` is the origin. | `false` |
| `rule` | [`KernelRule`](#advanced-functionality-different-kernel-rules) | Instantiated strategy to use to compute or approximate the kernel matrix. | `KernelRule()` |

***Notes:***

 * Selecting the correct `KernelType` depends strongly on the data.  A decent
   default is the [`GaussianKernel`](../core/kernels.md#gaussiankernel),
   although its bandwidth needs to be tuned to the data.  But
   [`GaussianKernel`](../core/kernels.md#gaussiankernel) may not always be the
   best: depending on the distribution and shape of the data, different kernels
   may yield extremely different results.

 * If [`LinearKernel`](../core/kernels.md#linearkernel) is used, then kernel PCA
   is equivalent to regular PCA and the [`PCA`](pca.md) class can be used
   instead for more efficient results.

### Applying transformations

 * `k,Apply(data, transformedData, newDimension)`
   - Apply kernel PCA to `data`, keeping only the top `newDimension`
     kernel eigendimensions.
   - The resulting transformed data is stored in `transformedData`, and will
     have size `newDimension` x `data.n_cols`.

 * `k.Apply(data, transformedData, eigval, eigvec, newDimension)`
   - Apply kernel PCA to the given `data`, keeping only the top `newDimension`
     kernel eigendimensions.
   - The resulting transformed data is stored in `transformedData`, and will
     have size `newDimension` x `data.n_cols`.
   - The top `newDimension` eigenvalues of the kernel matrix are stored
     descending order in the `eigval` vector, which will be set to length
     `newDimension`.
   - The top `newDimension` kernel eigenvectors associated with each kernel
     eigenvalue are stored in the matrix `eigvec`, which will be set to size
     `newDimension` x `data.n_cols`.
   - The `i`th kernel eigenvector can be obtained with `eigvec.col(i)` and its
     corresponding kernel eigenvalue is `eigval[i]`.

---

 * `k.Apply(data, transformedData, eigval)`
   - Apply kernel PCA to the given `data`, storing the result in
     `transformedData`, which will have size `data.n_cols` x `data.n_cols`.
   - Eigenvalues of the kernel matrix are stored in descending order in the
     vector `eigval`, which will be set to length `data.n_cols`.
   - If `data.n_cols` is large, consider one of the `Apply()` overloads below
     that takes a `newDimension` parameter, which results in a much smaller
     transformed data matrix.

 * `k.Apply(data, transformedData, eigval, eigvec)`
   - Apply kernel PCA to the given `data`, storing the result in
     `transformedData`, which will have size `data.n_cols` x `data.n_cols`.
   - Eigenvalues of the kernel matrix are stored in descending order in the
     vector `eigval`, which will be set to length `data.n_cols`.
   - Kernel matrix eigenvectors associated with each kernel eigenvalue are
     stored in the matrix `eigvec`, which will be set to size `data.n_cols` x
     `data.n_cols`.
   - The `i`th kernel eigenvector can be obtained with `eigvec.col(i)` and its
     corresponding kernel eigenvalue is `eigval[i]`.
   - If `data.n_cols` is large, consider one of the `Apply()` overloads below
     that takes a `newDimension` parameter, which results in a much smaller
     transformed data matrix.

#### `Apply()` Parameters:

| **name** | **type** | **description** |
|-----------|----------|----------|-----------------|
| `data` | [`arma::mat`](../matrices.md) | Input dataset to perform kernel PCA on. |
| `transformedData` | [`arma::mat`](../matrices.md) | Matrix to store transformed data into; will be filled with data of size `data.n_cols` x `data.n_cols`, or `newDimension` x `data.n_cols` if `newDimension` was specified. |
| `newDimension` | `size_t` | Desired new dimensionality of data (e.g. number of kernel eigenvectors to keep). |
| `eigval` | [`arma::vec&`](../matrices.md) | Vector to store kernel eigenvalues into (in descending order). |
| `eigvec` | [`arma::mat&`](../matrices.md) | Matrix to store kernel eigenvectors into. |

***Notes:***

 * Kernel PCA scales *quadratically* in the number of points (`data.n_cols`).
   If your dataset is large, `Apply()` may exhaust the system's memory or take a
   very long time.  In that case, the use of an approximate kernel matrix rule
   as the `KernelRule` template parameter may be helpful; see the
   [advanced functionality section](#advanced-functionality-different-kernel-rules).

 * Different element types can be used for `data` (e.g. `arma::fmat` or any
   dense matrix type implementing the Armadillo API).  If a different type is
   used for `data`, then the types of `transformedData`, `eigval`, and `eigvec`
   must match the given element type.  For example, if `data` is `arma::fmat`,
   then `transformedData` and `eigvec` must have type `arma::fmat`, and `eigval`
   must have type `arma::fvec`.

### Other functionality

 * `k.Kernel()` returns the `KernelType` that will be used for computing the
   kernel matrix.  If a `KernelType` was passed to the
   [constructor](#constructors), this is a copy of that object.

 * `k.CenterTransformedData()` returns a `bool` indicating whether or not the
   transformed data will be centered about the origin;
   `k.CenterTransformedData() = c` can be used to change the setting.

 * A `KernelPCA` object can be serialized with
   [`Save()` and `Load()`](../load_save.md#mlpack-models-and-objects).

### Simple examples

Use kernel PCA on the concentric circles dataset with a downstream logistic
regression classifier; print the classification results before and after kernel
PCA.

```c++
// Load the datasets.
arma::mat data;
mlpack::Load("http://datasets.mlpack.org/circles.csv", data, mlpack::Fatal);
arma::Row<size_t> labels;
mlpack::Load("http://datasets.mlpack.org/circles.labels.csv", labels,
    mlpack::Fatal);

// Perform kernel PCA on the dataset with default parameters.
mlpack::KernelPCA<mlpack::GaussianKernel> kpca;
arma::mat transformedData;
kpca.Apply(data, transformedData, 2);

// Now construct two classifiers: one on the original dataset, and one on the
// transformed dataset.
mlpack::LogisticRegression lr1, lr2;
lr1.Train(data, labels);
lr2.Train(transformedData, labels);

// Compute the correctness of each on the training data.  The preprocessed
// dataset will have perfect results because it has been mapped to a linearly
// separable space!
arma::Row<size_t> preds1, preds2;
lr1.Classify(data, preds1);
lr2.Classify(transformedData, preds2);

std::cout << "Logistic regression correct on original dataset:    "
    << arma::accu(preds1 == labels) << " of " << labels.n_elem << "."
    << std::endl;
std::cout << "Logistic regression correct on transformed dataset: "
    << arma::accu(preds2 == labels) << " of " << labels.n_elem << "."
    << std::endl;
```

---

Perform kernel PCA with the Gaussian kernel with a specified bandwidth on the
concentric circles dataset, and print some eigenvalues of the kernel matrix.

```c++
arma::mat data;
mlpack::Load("http://datasets.mlpack.org/circles.csv", data, mlpack::Fatal);

// Create the kernel with a bandwidth of 0.75, and then the kernel PCA object.
mlpack::GaussianKernel gk(0.75);
mlpack::KernelPCA<mlpack::GaussianKernel> kpca(gk);

// Transform the data, returning the eigenvalues and eigenvectors.
arma::mat transformedData, eigenvectors;
arma::vec eigenvalues;
kpca.Apply(data, transformedData, eigenvalues, eigenvectors);

// Print some eigenvalues and eigenvectors.
std::cout << "Total number of eigenvalues: " << eigenvalues.n_elem << "."
    << std::endl;
std::cout << "Size of eigenvectors matrix: " << eigenvectors.n_rows << " x "
    << eigenvectors.n_cols << "." << std::endl;
std::cout << "First five eigenvalues: " << eigenvalues.subvec(0, 4).t();
```

---

Perform kernel PCA on the cloud dataset using the cosine similarity as the
kernel function, mapping down to three dimensions.

```c++
arma::mat data;
mlpack::Load("http://datasets.mlpack.org/cloud.csv", data, mlpack::Fatal);

// Use the cosine similarity as the kernel function.
mlpack::KernelPCA<mlpack::CosineSimilarity> kpca;

arma::mat transformedData;
kpca.Apply(data, transformedData, 3 /* dimensions */);

// Print the first three points before and after.
for (size_t i = 0; i < 3; ++i)
{
  std::cout << "Point " << i << ":" << std::endl;
  std::cout << " - Before KPCA: " << data.col(i).t();
  std::cout << " - After KPCA:  " << transformedData.col(i).t();
}
```

---

Create a kernel PCA object and save it to disk for later usage.

```c++
mlpack::KernelPCA<mlpack::LaplacianKernel> kpca;
kpca.Kernel().Bandwidth(50.0);

// Save the object to `kpca.bin`.
mlpack::Save("kpca.bin", kpca, mlpack::Fatal);
```

---

Load a kernel PCA object from disk and perform kernel PCA on the satellite
dataset using the Laplacian kernel with a custom bandwidth, using 32-bit
floating point values to represent the data.

```c++
arma::fmat data;
mlpack::Load("http://datasets.mlpack.org/satellite.train.csv", data,
    mlpack::Fatal);

// Load a KernelPCA object from disk.  `kpca.bin` is expected to contain a
// KernelPCA object with a LaplacianKernel that has a bandwidth of 50.
mlpack::KernelPCA<mlpack::LaplacianKernel> kpca;
mlpack::Load("kpca.bin", kpca, mlpack::Fatal);

// Compute kernel PCA, keeping the top 3 dimensions and their eigenvalues and
// eigenvectors.
arma::fmat transformedData, eigenvectors;
arma::fvec eigenvalues;
kpca.Apply(data, transformedData, eigenvalues, eigenvectors, 3);

std::cout << "Transformed data size: " << transformedData.n_rows << " x "
    << transformedData.n_cols << "." << std::endl;
std::cout << "First three eigenvalues: " << eigenvalues.t();
```

---

### Advanced functionality: different kernel rules

The `KernelPCA` class has two template parameters that can be used for custom
behavior.  The full signature of the class is:

```
KernelPCA<KernelType, KernelRule>
```

 * `KernelType` is the type of kernel function to use when computing the kernel
   matrix.  There are many [mlpack kernels](../core/kernels.md) available for
   drop-in usage, and
   [implementing a custom kernel](../core/kernels.md#implement-a-custom-kernel)
   is quite straightforward.  Common choices of kernels include:
   - [`GaussianKernel`](../core/kernels.md#gaussiankernel)
   - [`CauchyKernel`](../core/kernels.md#cauchykernel)
   - [`LaplacianKernel`](../core/kernels.md#laplaciankernel)
   - [`PolynomialKernel`](../core/kernels.md#polynomialkernel)
   - [`SphericalKernel`](../core/kernels.md#sphericalkernel)

 * `KernelRule` specifies the strategy to be used to compute the kernel matrix.
   Because kernel PCA depends on the computation (or approximation) of the
   kernel matrix, and the kernel matrix has size `data.n_cols` x `data.n_cols`,
   kernel PCA scales very poorly to large datasets.  The use of a custom
   `KernelRule` can help mitigate this problem.
   - `NaiveKernelRule` (the default) simply computes the kernel matrix as a
     `data.n_cols` x `data.n_cols` matrix, by computing the kernel function
     between each pair of points.

   - `NystroemKernelRule<>` is a customizable approach based on
     [`NystroemMethod`](nystroem_method.md) that scales significantly better, by
     taking a small sample of points from the dataset and using a low-rank
     approximation of the sampled kernel matrix.
     * By default, 5 iterations of [`KMeans`](kmeans.md) is used to compute a
       sample of `0.1 * data.n_cols` centroids.
     * The `NystroemKernelRule<>` class is configurable; see
       [its documentation below](#nystroemkernelrule).

   - A custom kernel rule is straightforward to implement and must implement the
     following API:

```c++
class CustomKernelRule
{
  // A default constructor must be available.
  CustomKernelRule();

  // Compute or approximate the kernel matrix of `data`, using the given kernel
  // `kernel`.  The approximate eigenvalues and eigenvectors should be computed
  // and stored in `eigval` and `eigvec`, respectively, and the data should be
  // projected onto those eigenvectors and stored into `transformedData`.
  //
  // Note that a kernel rule does not need to compute *every* eigenvector and
  // eigenvalue of the kernel matrix, and may choose to only compute some.  That
  // is up to you as an implementer.
  //
  // Supposing that the class computes `r` eigenvalues and eigenvectors:
  //
  //  - the length of `eigval` is expected to be `r`;
  //  - the size of `eigvec` is expected to be `r` x `r`;
  //  - the size of `transformedData` is expected to be `data.n_cols` x `r`.
  //
  // `MatType` will be a dense matrix type that matches the Armadillo API (e.g.
  // `arma::mat`, and `VecType` will be the corresponding column vector type
  // (e.g.  `arma::vec`).
  template<typename MatType, typename VecType, typename KernelType>
  void ApplyKernelMatrix(const MatType& data,
                         MatType& transformedData,
                         VecType& eigval,
                         MatType& eigvec,
                         KernelType kernel);

  // The class must be serializable.
  //
  // If you never plan to serialize a KernelPCA that uses your kernel rule, you
  // can leave the function body empty.  Otherwise, see the cereal
  // documentation, or follow the example of other mlpack methods.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int version);
};
```

---

#### `NystroemKernelRule`

The `NystroemKernelRule<>` class uses the [`NystroemMethod`](nystroem_method.md)
class to compute a low-rank approximation to the kernel matrix using a small
number of samples from the dataset.  Different sample selection strategies can
be configured via template parameter, and the size of the sample can be
configured as a constructor parameter.

 * `NystroemKernelRule<KMeansSelection<>>` or `NystroemKernelRule<>` can be used
   to compute samples as centroids of a few iterations of [`KMeans`](kmeans.md)
   clustering.
    - The [`KMeansSelection<>`](nystroem_method.md#kmeansselection) class has
      two optional template parameters:
      `KMeansSelection<KMeansType, numIterations>`.
    - `KMeansType` can be any templated version of the [`KMeans`](kmeans.md)
      class.
    - `numIterations` is a `size_t` representing how many iterations of k-means
      to run.

 * `NystroemKernelRule<RandomSelection>` will select points randomly from the
   dataset.  (See also the
   [`RandomSelection`](nystroem_method.md#randomselection) documentation.)

 * `NystroemKernelRule<OrderedSelection>` will select the first points from the
   dataset.  (See also the
   [`OrderedSelection`](nystroem_method.md#orderedselection) documentation.)

 * A custom selection strategy can be implemented; the requirements are
   described in the
   [`NystroemMethod`](nystroem_method.md#custom-pointselectionpolicy-classes)
   documentation.

---

A `NystroemKernelRule<>` object has the following two constructors:

 * `n = NystroemKernelRule(datasetPct=0.01)`
 * `n = NystroemKernelRule<SelectionStrategy>(datasetPct=0.01)`
   - Construct a `NystroemKernelRule`, optionally with a specific
     `SelectionStrategy`.
   - The given percentage of the dataset size will be used as the sample size.

 * `n = NystroemKernelRule(numPoints)`
 * `n = NystroemKernelRule<SelectionStrategy>(numPoints)`
   - Construct a `NystroemKernelRule`, optionally with a specific
     `SelectionStrategy`.
   - The given number of points will be used as the sample size.

### Advanced examples

These examples use custom `KernelRule`s to approximate the kernel matrix and
scale kernel PCA to larger datasets.

---

Perform kernel PCA on the cloud dataset, using the
[`NystroemKernelRule`](#nystroemkernelrule) to approximate the kernel matrix.

```c++
arma::mat data;
mlpack::Load("http://datasets.mlpack.org/cloud.csv", data, mlpack::Fatal);

// Use the Nystroem method with default parameters to approximate the kernel
// matrix.
mlpack::KernelPCA<mlpack::GaussianKernel, mlpack::NystroemKernelRule<> > kpca;
kpca.Kernel().Bandwidth(250.0);

arma::mat transformedData;
kpca.Apply(data, transformedData, 3 /* dimensions */);

// Print the first five data points.
std::cout << "First five data points:" << std::endl;
for (size_t i = 0; i < 5; ++i)
{
  std::cout << " - Point " << i << ": " << transformedData.col(i).t();
}
```

---

Perform kernel PCA on the corel dataset, using 100 randomly selected points with
the [`NystroemKernelRule`](#nystroemkernelrule) to approximate the kernel
matrix.

```c++
arma::mat data;
mlpack::Load("http://datasets.mlpack.org/corel-histogram.csv", data,
    mlpack::Fatal);

// Instantiate the Nystroem kernel rule so that we can set the number of points
// to use to 100 instead of the default of 1% of the dataset.
mlpack::NystroemKernelRule<mlpack::RandomSelection> nkr(100);
mlpack::KernelPCA<mlpack::GaussianKernel,
                  mlpack::NystroemKernelRule<mlpack::RandomSelection>> kpca(
    mlpack::GaussianKernel(), true, nkr);

// Transform the data, keep the top 5 dimensions, and get the eigenvalues.
arma::mat transformedData, eigenvectors;
arma::vec eigenvalues;
kpca.Apply(data, transformedData, eigenvalues, eigenvectors, 5);

// Print the eigenvalues.
std::cout << "First 5 eigenvalues of approximated kernel matrix: "
    << eigenvalues.t();
```

---

Perform kernel PCA on the corel dataset using 32-bit floats to represent the
data.  Use [`ElkanKMeans`](kmeans.md#lloydsteptype) to efficiently perform 10
iterations of k-means, producing 50 centroids to approximate the data.

```c++
arma::fmat data;
mlpack::Load("http://datasets.mlpack.org/corel-histogram.csv", data,
    mlpack::Fatal);

// Instantiate the Nystroem kernel rule to use k-means with the Elkan strategy,
// computing a number of centroids equal to 2% of the data points.
using KernelRuleType = mlpack::NystroemKernelRule<mlpack::KMeansSelection<
    mlpack::KMeans<mlpack::EuclideanDistance,
                   mlpack::SampleInitialization,
                   mlpack::MaxVarianceNewCluster,
                   mlpack::ElkanKMeans>, 10 /* iterations */>>;
KernelRuleType nkr(0.02);
mlpack::KernelPCA<mlpack::GaussianKernel, KernelRuleType> kpca(
    mlpack::GaussianKernel(), true, nkr);

// Transform the data, keep the top 5 dimensions, and get the eigenvalues.
arma::fmat transformedData, eigenvectors;
arma::fvec eigenvalues;
kpca.Apply(data, transformedData, eigenvalues, eigenvectors, 5);

// Print the eigenvalues.
std::cout << "First 5 eigenvalues of approximated kernel matrix: "
    << eigenvalues.t();
```
