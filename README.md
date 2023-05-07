Download Link: https://assignmentchef.com/product/solved-ml-homework1-the-learning-problem
<br>
<strong>The Learning Problem</strong>

<ol>

 <li>Which of the following problem is suited for machine learning if there is assumed to be enough associated data? Choose the correct answer; explain how you can possibly use machine learning to solve it.

  <ul>

   <li>predicting the winning number of the next invoice lottery</li>

   <li>calculating the average score of 500 students</li>

   <li>identifying the exact minimal spanning tree of a graph</li>

   <li>ranking mango images by the quality of the mangoes</li>

   <li>none of the other choices</li>

  </ul></li>

 <li>Which of the following describes an machine learning approach to build a system for spam detection? Choose the correct answer; explain briefly why you think other choices are <em>not </em>machine learning.

  <ul>

   <li>flip 3 fair coins; classify the email as a spam iff at least 2 of them are heads</li>

   <li>forward the email to 3 humans; classify the email as a spam iff at least 2 of them believe so</li>

   <li>produce a list of words for spams by 3 humans; classify the email as a spam iff the email contains more than 10 words from the list</li>

   <li>get a data set that contains spams and non-spams, for all words in the data set, let the machine calculate the ratio of spams per word; produce a list of words that appear more than 5 times and are of the highest 20% ratio; classify the email as a spam iff the email contains more than 10 words from the list</li>

   <li>get a data set that contains spams and non-spams, for all words in the data set, let the machine decide its “spam score”; sum the score up for each email; let the machine optimize a threshold that achieves the best precision of spam detection; classify the email as a spam iff the email is of score more than the threshold</li>

  </ul></li>

</ol>

<strong>Perceptron Learning Algorithm</strong>

Next, we will play with multiple variations to the Perceptron Learning Algorithm (PLA).

<ol start="3">

 <li>Short scales down all <strong>x</strong><em><sub>n </sub></em>(including the <em>x</em><sub>0 </sub>within) linearly by a factor of 4 before running PLA. How does the worst-case speed of PLA (in terms of the bound on page 16 of lecture 2) change after scaling? Choose the correct answer; explain your answer.

  <ul>

   <li>4 times smaller (i.e. faster)</li>

   <li>2 times smaller</li>

  </ul></li>

</ol>

√

<ul>

 <li>2 times smaller</li>

 <li>unchanged</li>

</ul>

√

<ul>

 <li>2 times larger (i.e. slower)</li>

</ul>

<ol start="4">

 <li>The scaling in the previous problem is equivalent to inserting a “learning rate” <em>η </em>to the PLA update rule</li>

</ol>

<strong>w</strong><em>t</em>+1 ← <strong>w</strong><em>t </em>+ <em>η </em>· <em>y</em><em>n</em>(<em>t</em>)<strong>x</strong><em>n</em>(<em>t</em>)

with. In fact, we do not need to use a fixed <em>η</em>. Let <em>η<sub>t </sub></em>denote the learning rate in the <em>t</em>-th iteration; that is, let PLA update <strong>w</strong><em><sub>t </sub></em>by

<strong>w</strong><em>t</em>+1 ← <strong>w</strong><em>t </em>+ <em>η</em><em>t </em>· <em>y</em><em>n</em>(<em>t</em>)<strong>x</strong><em>n</em>(<em>t</em>)

whenever (<strong>x</strong><em><sub>n</sub></em><sub>(<em>t</em>)</sub><em>,y<sub>n</sub></em><sub>(<em>t</em>)</sub>) is not correctly classified by <strong>w</strong><em><sub>t</sub></em>. Dr. Adaptive decides to set so

“longer” <strong>x</strong><em><sub>n</sub></em><sub>(<em>t</em>) </sub>will not affect <strong>w</strong><em><sub>t </sub></em>too much. Let

<em>,</em>

which can be viewed as a “normalized” version of the <em>ρ </em>on page 16 of lecture 2. The bound on the same page then becomes ˆ<em>ρ</em><sup>−<em>p </em></sup>after using this adaptive <em>η<sub>t</sub></em>. What is <em>p</em>? Choose the correct answer; explain your answer.

<ul>

 <li>0</li>

</ul>

<h1>[b] 1</h1>

<h1>[c] 2</h1>

<h1>[d] 4</h1>

<strong>[e] </strong>8

<ol start="5">

 <li>Another possibility of setting <em>η<sub>t </sub></em>is to consider how negative <em>y<sub>n</sub></em><sub>(<em>t</em>)</sub><strong>w</strong><em><sub>t</sub><sup>T</sup></em><strong>x</strong><em><sub>n</sub></em><sub>(<em>t</em>) </sub>is, and try to make</li>

</ol>

0; that is, let <strong>w</strong><em><sub>t</sub></em><sub>+1 </sub>correctly classify (<strong>x</strong><em><sub>n</sub></em><sub>(<em>t</em>)</sub><em>,y<sub>n</sub></em><sub>(<em>t</em>)</sub>). Which of the following update

rules make   0? Choose the correct answer; explain your answer.

<strong>[a] w</strong><em>t</em>+1 ← <strong>w</strong><em>t </em>+ 2 · <em>y</em><em>n</em>(<em>t</em>)<strong>x</strong><em>n</em>(<em>t</em>)

<ol start="6">

 <li>Separate decides to use one of the update rules in the previous problem for PLA. When the data set is linear separable, how many choices in the previous problem ensures halting with a “perfect line”? Choose the correct answer; explain the reason behind each halting case.</li>

</ol>

<h1>[a] 1</h1>

<ul>

 <li>2</li>

 <li>3</li>

 <li>4</li>

 <li>5</li>

</ul>

<strong>Types of Learning</strong>

<ol start="7">

 <li>One shared technique between the famous AlphaGo, AlphaGo Zero, and AlphaStar is called selfpracticing: learning to play the game by practicing with itself and getting the feedback from the “judge” environment. What best describes the learning problem behind self-practicing? Choose the correct answer; explain your answer.

  <ul>

   <li>human learning</li>

   <li>unsupervised learning</li>

   <li>semi-supervised learning</li>

   <li>supervised learning</li>

   <li>reinforcement learning</li>

  </ul></li>

 <li>Consider formulating a learning problem for building a self-driving car. First, we gather a training data set that consists of 100 hours of video that contains the view in front of a car, and records about how the human behind the wheel acted with physically constrained choices like steering, braking, and signaling-before-turning. We also gather another 100 hours of videos from 1126 more cars without the human records. The learning algorithm is expected to learn from all the videos to obtain a hypothesis that imitates the human actions well. What learning problem best matches the description above? Choose the correct answer; explain your answer.

  <ul>

   <li>regression, unsupervised learning, active learning, concrete features</li>

   <li>structured learning, semi-supervised learning, batch learning, raw features</li>

   <li>structured learning, supervised learning, batch learning, concrete features</li>

   <li>regression, reinforcement learning, batch learning, concrete features</li>

   <li>structured learning, supervised learning, online learning, concrete features</li>

  </ul></li>

</ol>

(<em>We are definitely not hinting that you should build a self-driving car this way. &#x1f600; )</em>

<strong>Off-Training-Set Error</strong>

As discussed on page 5 of lecture 4, what we really care about is whether <em>g </em>≈ <em>f outside </em>D. For a set of “universe” examples U with D ⊂ U, the error <em>outside </em>D is typically called the Off-Training-Set (OTS) error

1              X

<em>E</em><sub>ots</sub>(<em>h</em>) =    <em>                     h</em>(<strong>x</strong>) 6= <em>y .</em>

|U  D| (<strong>x</strong><em>,y</em>)∈UD J                   K

<ol start="9">

 <li>Consider U with 6 examples</li>

</ol>

Run the process of choosing any three examples from U as D, and learn a perceptron hypothesis (say, with PLA, or any of your “human learning” algorithm) to achieve <em>E</em><sub>in</sub>(<em>g</em>) = 0 on D. Then, evaluate <em>g </em>outside D. What is the smallest and largest <em>E</em><sub>ots</sub>(<em>g</em>)? Choose the correct answer; explain your answer.

<strong>[e] </strong>(0<em>,</em>1)

<strong>Hoeffding Inequality</strong>

<ol start="10">

 <li>Suppose you are given a biased coin with one side coming up with probability . How many times do you need to toss the coin to find out the more probable side with probability at least 1−<em>δ </em>using the Hoeffding’s Inequality mentioned in page 10 of lecture 4? Choose the correct answer; explain your answer. (<em>Hint: There are multiple versions of Hoeffding’s inequality. Please use the version in the lecture, albeit slightly loose, for answering this question. The </em>log <em>here is </em>log<em><sub>e</sub>.</em>)</li>

</ol>

<strong>Bad Data</strong>

<ol start="11">

 <li>Consider <strong>x </strong>= [<em>x</em><sub>1</sub><em>,x</em><sub>2</sub>]<em><sup>T </sup></em>∈ R<sup>2</sup>, a target function <em>f</em>(<strong>x</strong>) = sign(<em>x</em><sub>1</sub>), a hypothesis <em>h</em><sub>1</sub>(<strong>x</strong>) = sign(2<em>x</em><sub>1</sub>−<em>x</em><sub>2</sub>), and another hypothesis <em>h</em><sub>2</sub>(<strong>x</strong>) = sign(<em>x</em><sub>2</sub>). When drawing 5 examples independently and uniformly within [−1<em>,</em>+1] × [−1<em>,</em>+1] as D, what is the probability that we get 5 examples (<strong>x</strong><em><sub>n</sub>,f</em>(<strong>x</strong><em><sub>n</sub></em>)) such that <em>E</em><sub>in</sub>(<em>h</em><sub>2</sub>) = 0? Choose the correct answer; explain your answer. (<em>Note: This is one of the BAD-data cases for h</em><sub>2 </sub><em>where E</em><sub>in</sub>(<em>h</em><sub>2</sub>) <em>is far from E</em><sub>out</sub>(<em>h</em><sub>2</sub>).)</li>

</ol>

<h1>[a] 0</h1>

<strong>[b] </strong><strong>[c] </strong>

<strong>[d] </strong>

<strong>[e] </strong>1

<ol start="12">

 <li>Following the setting of the previous problem, what is the probability that we get 5 examples such that <em>E</em><sub>in</sub>(<em>h</em><sub>2</sub>) = <em>E</em><sub>in</sub>(<em>h</em><sub>1</sub>), including both the zero and non-zero <em>E</em><sub>in </sub>cases? Choose the correct answer; explain your answer. (<em>Note: This is one of the BAD-data cases where we cannot distinguish the better-E</em><sub>out </sub><em>hypothesis h</em><sub>1 </sub><em>from the worse hypothesis h</em><sub>2</sub><em>.</em>)</li>

</ol>

<h1>[a]</h1>

<strong>[b] </strong><strong>[c] </strong><strong>[d] </strong>

<strong>[e] </strong>

<ol start="13">

 <li>According to page 22 of lecture 4, for a hypothesis set H,</li>

</ol>

BAD D for H ⇐⇒ ∃<em>h </em>∈ H s.t.

Let <strong>x </strong>= [<em>x</em><sub>1</sub><em>,x</em><sub>2</sub><em>,</em>··· <em>,x<sub>d</sub></em>]<em><sup>T </sup></em>∈ R<em><sup>d </sup></em>with <em>d &gt; </em>1. Consider a binary classification target with Y = {+1<em>,</em>−1} and a hypothesis set H with 2<em>d </em>hypotheses <em>h</em><sub>1</sub><em>,</em>··· <em>,h</em><sub>2<em>d</em></sub>.

<ul>

 <li>For <em>i </em>= 1<em>,</em>·· <em>,d</em>, <em>h<sub>i</sub></em>(<strong>x</strong>) = sign(<em>x<sub>i</sub></em>).</li>

 <li>For <em>i </em>= <em>d </em>+ 1<em>,</em>·· <em>,</em>2<em>d</em>, <em>h<sub>i</sub></em>(<strong>x</strong>) = −sign(<em>x<sub>i</sub></em><sub>−<em>d</em></sub>).</li>

</ul>

Extend the Hoeffding’s Inequality mentioned in page 10 of lecture 4 with a proper union bound. Then, for any given <em>N </em>and , what is the smallest <em>C </em>that makes this inequality true?

P[BAD<em>.</em>

Choose the correct answer; explain your answer.

<ul>

 <li><em>C </em>= 1 <strong>[b] </strong><em>C </em>= <em>d</em></li>

 <li><em>C </em>= 2<em>d</em></li>

 <li><em>C </em>= 4<em>d </em><strong>[e] </strong><em>C </em>= ∞</li>

</ul>

<strong>Multiple-Bin Sampling</strong>

<ol start="14">

 <li>We then illustrate what happens with multiple-bin sampling with an experiment that use a dice (instead of a marble) to bind the six faces together. Please note that the dice is not meant to be thrown for random experiments. The probability below only refers to drawing the dices from the bag. Try to view each number as a <em>hypothesis</em>, and each dice as an <em>example </em>in our multiple-bin scenario. You can see that no single number is always green—that is, <em>E</em><sub>out </sub>of each hypothesis is always non-zero. In the next two problems, we are essentially asking you to calculate the probability of getting <em>E</em><sub>in</sub>(<em>h</em><sub>3</sub>) = 0, and the probability of the minimum <em>E</em><sub>in</sub>(<em>h<sub>i</sub></em>) = 0.</li>

</ol>

Consider four kinds of dice in a bag, with the same (super large) quantity for each kind.

<ul>

 <li>A: all even numbers are colored green, all odd numbers are colored orange</li>

 <li>B: (2<em>,</em>3<em>,</em>4) are colored green, others are colored orange</li>

 <li>C: the number 6 is colored green, all other numbers are colored orange</li>

 <li>D: all primes are colored green, others are colored orange</li>

</ul>

If we draw 5 dices independently from the bag, which combination is of the same probability as getting five green 3’s? Choose the correct answer; explain your answer.

<ul>

 <li>five green 1’s</li>

 <li>five orange 2’s</li>

 <li>five green 2’s</li>

 <li>five green 4’s <strong>[e] </strong>five green 5’s</li>

</ul>

<ol start="15">

 <li>Following the previous problem, if we draw 5 dices independently from the bag, what is the probability that we get <em>some number </em>that is purely green? Choose the correct answer; explain your answer.</li>

</ol>

<strong>[a] </strong>

<strong>[b] </strong><strong>[c] </strong><strong>[d] </strong>

<strong>[e] </strong>

<strong>Experiments with Perceptron Learning Algorithm</strong>

Next, we use an artificial data set to study PLA. The data set with <em>N </em>= 100 examples is in

http://www.csie.ntu.edu.tw/~htlin/course/ml20fall/hw1/hw1_train.dat

Each line of the data set contains one (<strong>x</strong><em><sub>n</sub>,y<sub>n</sub></em>) with <strong>x</strong><em><sub>n </sub></em>∈ R<sup>10</sup>. The first 10 numbers of the line contains the components of <strong>x</strong><em><sub>n </sub></em>orderly, the last number is <em>y<sub>n</sub></em>. Please initialize your algorithm with <strong>w </strong>= <strong>0 </strong>and take sign(0) as −1.

<ol start="16">

 <li>(*) Please first follow page 4 of lecture 2, and add <em>x</em><sub>0 </sub>= 1 to every <strong>x</strong><em><sub>n</sub></em>. Implement a version of PLA that randomly picks an example (<strong>x</strong><em><sub>n</sub>,y<sub>n</sub></em>) in every iteration, and updates <strong>w</strong><em><sub>t </sub></em>if and only if <strong>w</strong><em><sub>t </sub></em>is incorrect on the example. Note that the random picking can be simply implemented <em>with replacement</em>—that is, the same example can be picked multiple times, even consecutively. Stop updating and return <strong>w</strong><em><sub>t </sub></em>as <strong>w</strong><sub>PLA </sub>if <strong>w</strong><em><sub>t </sub></em>is correct consecutively after checking 5<em>N </em>randomly-picked examples.</li>

</ol>

<em>Hint: (1) The update procedure described above is equivalent to the procedure of gathering all the incorrect examples first and then randomly picking an example among the incorrect ones. But the description above is usually much easier to implement. (2) The stopping criterion above is a randomized, more efficient implementation of checking whether </em><strong>w</strong><em><sub>t </sub>makes no mistakes on the data set.</em>

Repeat your experiment for 1000 times, each with a different random seed. What is the median number of updates before the algorithm returns <strong>w</strong><sub>PLA</sub>? Choose the closest value.

<ul>

 <li>8</li>

 <li>11</li>

 <li>14</li>

 <li>17</li>

 <li>20</li>

</ul>

<ol start="17">

 <li>(*) Among all the <em>w</em><sub>0 </sub>(the zero-th component of <strong>w</strong><sub>PLA</sub>) obtained from the 1000 experiments above, what is the median? Choose the closest value.

  <ul>

   <li>-10</li>

   <li>-5 <strong>[c] </strong>0</li>

   <li>5</li>

   <li>10</li>

  </ul></li>

 <li>(*) Set <em>x</em><sub>0 </sub>= 10 to every <strong>x</strong><em><sub>n </sub></em>instead of <em>x</em><sub>0 </sub>= 1, and repeat the 1000 experiments above. What is the median number of updates before the algorithm returns <strong>w</strong><sub>PLA</sub>? Choose the closest value.

  <ul>

   <li>8</li>

   <li>11</li>

   <li>14</li>

   <li>17</li>

   <li>20</li>

  </ul></li>

 <li>(*) Set <em>x</em><sub>0 </sub>= 0 to every <strong>x</strong><em><sub>n </sub></em>instead of <em>x</em><sub>0 </sub>= 1. This equivalently means not adding any <em>x</em><sub>0</sub>, and you will get a separating hyperplane that passes the origin. Repeat the 1000 experiments above. What is the median number of updates before the algorithm returns <strong>w</strong><sub>PLA</sub>?

  <ul>

   <li>8</li>

   <li>11</li>

   <li>14</li>

   <li>17</li>

   <li>20</li>

  </ul></li>

 <li>(*) Now, in addition to setting <em>x</em><sub>0 </sub>= 0 to every <strong>x</strong><em><sub>n</sub></em>, scale down each <strong>x</strong><em><sub>n </sub></em>by 4. Repeat the 1000 experiments above. What is the median number of updates before the algorithm returns <strong>w</strong><sub>PLA</sub>? Choose the closest value.

  <ul>

   <li>8</li>

   <li>11</li>

   <li>14</li>

   <li>17</li>

   <li>20</li>

  </ul></li>

</ol>