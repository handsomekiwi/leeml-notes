## 深度強化學習淺析

2015年2月的時候，google在nature上發了一篇用reinforcement learning 的方法來玩akari的小遊戲，然後痛鞭人類

2016的春天，又有大家都耳熟能詳的alpha go，也是可以痛鞭人類

David Silver 說 AI 就是 Reinforcement Learning加Deep Learning
Deep Reinforcement Learning : AI = RL + DL

## 強化學習的應用場景

在Reinforcement Learning裡面會有一個Agent跟一個Environment。這個Agent會有Observation看到世界種種變化，這個Observation又叫做State，這個State指的是環境的狀態，也就是你的machine所看到的東西。所以在這個Reinforcement Learning領域才會有這個XXX做法，我們的state能夠觀察到一部分的情況，機器沒有辦法看到環境所有的狀態，所以才會有這個partial of state 這個想法，這個state其實就是Observation。machine會做一些事情，它做的事情叫做Action，Action會影響環境，會跟環境產生一些互動。因為它對環境造成的一些影響，它會得到Reward，這個Reward告訴它，它的影響是好的還是不好的。如下圖

![39_1](./res/chapter39_1.png)

舉個例子，比如機器看到一杯水，然後它就take一個action，這個action把水打翻了，Environment就會得到一個negative的reward，告訴它不要這樣做，它就得到一個負向的reward。在Reinforcement Learning這些動作都是連續的，因為水被打翻了，接下來它看到的就是水被打翻的狀態，它會take另外一個action，決定把它擦乾淨，Environment覺得它做得很對，就給它一個正向的reward。機器生來的目標就是要去學習採取那些ation，可以讓maximize reward maximize 。

接著，以alpha go為例子，一開始machine的Observation是棋盤，棋盤可以用一個19*19的矩陣來描述，接下來，它要take一個action，這個action就是落子的位置。落子在不同的位置就會引起對手的不同反應，對手下一個子，Agent的Observation就變了。Agent看到另外一個Observation後，就要決定它的action，再take一個action，落子在另外一個位置。用機器下圍棋就是這麼個回事。在圍棋這個case裡面，還是一個蠻難的Reinforcement Learning，在多數的時候，你得到的reward都是0，落子下去通常什麼事情也沒發生這樣子。只有在你贏了，得到reward是1，如果輸了，得到reward是-1。Reinforcement Learning困難的地方就是有時候你的reward是sparse的，只有倒數幾步才有reward。即在只有少數的action 有reward的情況下去挖掘正確的action。

對於machine來說，它要怎麼學習下圍棋呢，就是找一某個對手一直下下，有時候輸有時候贏，它就是調整Observation和action之間的關係，調整model讓它得到的reward可以被maximize。

## 監督 v.s. 強化 

我們可以比較下下圍棋採用Supervised 和Reinforcement 有什麼區別。如果是Supervised 你就是告訴機器說看到什麼樣的態勢就落在指定的位置。Supervised不足的地方就是具體態勢下落在哪個地方是最好的，其實人也不知道，因此不太容易做Supervised。用Supervised就是machine從老師那學，老師說下哪就下哪。如果是Reinforcement 呢，就是讓機器找一個對手不斷下下，贏了就獲得正的reward，沒有人告訴它之前哪幾步下法是好的，它要自己去試，去學習。Reinforcement 是從過去的經驗去學習，沒有老師告訴它什麼是好的，什麼是不好的，machine要自己想辦法，其實在做Reinforcement 這個task裡面，machine需要大量的training，可以兩個machine互相下。alpha Go 是先做Supervised Learning，做得不錯再繼續做Reinforcement Learning。

## 應用舉例

### 學習一個chat-bot

 Reinforcement Learning 也可以被用在Learning a chat-bot。chat-bot 是seq2seq，input 就是一句話，output 就是機器的回答。

 如果採用Supervised ，就是告訴機器有人跟你說“hello”，你就回答“hi”。如果有人跟你說“bye bye”，你就要說“good bye”。

如果是Reinforcement  Learning 就是讓機器胡亂去跟人講話，講講，人就生氣了，machine就知道一句話可能講得不太好。不過沒人告訴它哪一句話講得不好，它要自己去發掘這件事情。

![39_2](./res/chapter39_2.png)


這個想法聽起來很crazy，但是真正有chat-bot是這樣做的，這個怎麼做呢？因為你要讓machine不斷跟人講話，看到人生氣後進行調整，去學怎麼跟人對話，這個過程比較漫長，可能得好幾百萬人對話之後才能學會。這個不太現實，那麼怎麼辦呢，就用Alpha Go的方式，Learning 兩個agent，然後讓它們互講的方式。

![39_3](./res/chapter39_3.png)

兩個chat-bot互相對話，對話之後有人要告訴它們它們講得好還是不好。在圍棋裡比較簡單，輸贏是比較明確的，對話的話就比較麻煩，你可以讓兩個machine進行無數輪互相對話，問題是你不知道它們這聊天聊得好還是不好，這是一個待解決問題。現有的方式是制定幾條規則，如果講得好就給它positive reward ，講得不好就給它negative reward，好不好由人主觀決定，然後machine就從它的reward中去學說它要怎麼講才是好。後續可能會有人用GAN的方式去學chat-bot。通過discriminator判斷是否像人對話，兩個agent就會想騙過discriminator，即用discriminator自動認出給reward的方式。
Reinforcement  Learning 有很多應用，尤其是人也不知道怎麼做的場景非常適合。

## 交互搜索

讓machine學會做Interactive retrieval，意思就是說有一個搜尋系統，能夠跟user進行信息確認的方式，從而搜尋到user所需要的信息。直接返回user所需信息，它會得到一個positive reward，然後每問一個問題，都會得到一個negative reward。

![39_4](./res/chapter39_4.png)

### 更多應用

Reinforcement  Learning 還有很多應用，比如開個直升機，開個無人車呀，也有通過deepmind幫助谷歌節電，也有文本生成等。現在Reinforcement  Learning最常用的場景是電玩。現在有現成的environment，比如Gym,Universe。讓machine 用Reinforcement  Learning來玩遊戲，跟人一樣，它看到的東西就是一幅畫面，就是pixel，然後看到畫面，它要做什麼事情它自己決定，並不是寫程序告訴它說你看到這個東西要做什麼。需要它自己去學出來。

### 例子:玩視頻遊戲

- Space invader

	這個遊戲界面如下
	![39_5](./res/chapter39_5.png)
	遊戲的終止條件時當所有的外星人被消滅或者你的太空飛船被摧毀。

	這個遊戲裡面，你可以take的actions有三個，可以左右移動跟開火。怎麼玩呢，machine會看到一個observation，這個observation就是一幕畫面。一開始machine看到一個observation $s_1$，這個$s_1$其實就是一個matrix，因為它有顏色，所以是一個三維的pixel。machine看到這個畫面以後，就要決定它take什麼action，現在只有三個action可以選擇。比如它take 往右移。每次machine take一個action以後，它會得到一個reward，這個reward就是左上角的分數。往右移不會得到任何的reward，所以得到的reward $r_1 = 0$，machine 的action會影響環境，所以machine看到的observation就不一樣了。現在observation為$s_2$，machine自己往右移了，同時外星人也有點變化了，這個跟machine的action是沒有關係的，有時候環境會有一些隨機變化，跟machine無關。machine看到$s_2$之後就要決定它要take哪個action，假設它決定要射擊並成功的殺了一隻外星人，就會得到一個reward，發現殺不同的外星人，得到的分數是不一樣的。假設殺了一隻5分的外星人，這個observation就變了，少了一隻外星人。這個過程會一直進行下去，直到machine的reward $r_T$進入另一個step，這個step是一個terminal step，它會讓遊戲結束，在這個遊戲背景裡面就是你被殺死。可能這個machine往左移，不小心碰到alien的子彈，就死了，遊戲就結束了。從這個遊戲的開始到結束，就是一個episode，machine要做的事情就是不斷的玩這個遊戲，學習怎麼在一個episode裡面怎麼去maximize reward，maximize它在所有的episode可以得到的total reward。在死之前殺最多的外星人同時要閃避子彈，讓自己不會被殺死。

	![39_6](./res/chapter39_6.png)

## 強化學習的難點

那麼Reinforcement  Learning的難點在哪裡呢？它有兩個難點

- Reward delay

	第一個難點是，reward出現往往會存在delay，比如在space invader裡面只有開火才會得到reward，但是如果machine只知道開火以後就會得到reward，最後learn出來的結果就是它只會亂開火。對它來說，往左往右移沒有任何reward。事實上，往左往右這些moving，它對開火是否能夠得到reward是有關鍵影響的。雖然這些往左往右的action，本身沒有辦法讓你得到任何reward，但它幫助你在未來得到reward，就像規劃未來一樣，machine需要有這種遠見，要有這種visual，才能把電玩玩好。在下圍棋裡面，有時候也是一樣的，短期的犧牲可以換來最好的結果。

- Agent's actions affect the subsequent data it receives

	Agent採取行動後會影響之後它所看到的東西，所以Agent要學會去探索這個世界。比如說在這個space invader裡面，Agent只知道往左往右移，它不知道開火會得到reward，也不會試著擊殺最上面的外星人，就不會知道擊殺這個東西可以得到很高的reward，所以要讓machine去explore它沒有做過的行為，這個行為可能會有好的結果也會有壞的結果。但是探索沒有做過的行為在Reinforcement  Learning裡面也是一種重要的行為。

## 強化學習的方法

Reinforcement  Learning 的方法分成兩大塊，一個是Policy-based的方法，另一個是Valued-based的方法。先有Valued-based的方法，再有Policy-based的方法。在Policy-based的方法裡面，會learn一個負責做事的Actor，在Valued-based的方法會learn一個不做事的Critic，專門批評不做事的人。我們要把Actor和Critic加起來叫做Actor+Critic的方法。

![39_7](./res/chapter39_7.png)

現在最強的方法就是Asynchronous Advantage Actor-Critic(A3C)。Alpha Go是各種方法大雜燴，有Policy-based的方法，有Valued-based的方法，有model-based的方法。下面是一些學習deep Reinforcement  Learning的資料

- Textbook: Reinforcement Learning: An Introduction
https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html
- Lectures of David Silver
http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html (10 lectures, 1:30 each)
http://videolectures.net/rldm2015_silver_reinforcement_learning/ (Deep Reinforcement Learning )
- Lectures of John Schulman
https://youtu.be/aUrX-rP_ss4

### Policy-based 方法

先來看看怎麼學一個Actor，所謂的Actor是什麼呢?我們之前講過，Machine Learning 就是找一個Function，Reinforcement Learning也是Machine Learning 的一種，所以要做的事情也是找Function。這個Function就是所謂的魔術發現，Actor就是一個Function。這個Function的input就是Machine看到的observation，它的output就是Machine要採取的Action。我們要透過reward來幫我們找這個best Function。

![39_8](./res/chapter39_8.png)

找個這個Function有三個步驟：

- Neural Network as Actor

	第一個步驟就是決定你的Function長什麼樣子，假設你的Function是一個Neural Network，就是一個deep learning。

	![39_9](./res/chapter39_9.png)

	如果Neural Network作為一個Actor，這個Neural Network的輸入就是observation，可以通過一個vector或者一個matrix 來描述。output就是你現在可以採取的action。舉個例子，Neural Network作為一個Actor，inpiut是一張image，output就是你現在有幾個可以採取的action，output就有幾個dimension。假設我們在玩Space invader，output就是可能採取的action左移、右移和開火，這樣output就有三個dimension分別代表了左移、右移和開火。

	![39_10](./res/chapter39_10.png)

	這個Neural Network怎麼決定這個Actor要採取哪個action呢？通常的做法是這樣，你把這個image丟到Neural Network裡面去，它就會告訴你說每個output dimension所對應的分數，可以採取分數最高的action，比如說left。用這個Neural Network來做Actor有什麼好處，Neural Network是可以舉一反三的，可能有些畫面Machine 從來沒有看過，但是Neural Network的特性，你給它一個image，Neural Network吐出一個output。所以就算是它沒看過的東西，它output可以得到一個合理的結果。Neural Network就是比較generous。

- Goodnedd of Actor

	第二步驟就是，我們要決定一個Actor的好壞。在Supervised learning中，我們是怎樣決定一個Function的好壞呢？舉個Training Example例子來說，我們把圖片扔進去，看它的結果和target是否像，如果越像的話這個Function就會越好，我們會一個loss，然後計算每個example的loss，我們要找一個參數去minimize這個參數。

	![39_11](./res/chapter39_11.png)

	在Reinforcement Learning裡面，一個Actor的好壞的定義是非常類似的。假設我們現在有一個Actor，這個Actor就是一個Neural Network，Neural Network的參數是$\mathbf{\theta}$，即一個Actor可以表示為$\pi_\theta(s)$，它的input就是Mechine看到的observation。那怎麼知道一個Actor表現得好還是不好呢？我們讓這個Actor實際的去玩一個遊戲，玩完遊戲得到的total reward為 $R_\theta=\sum_{t=1}^Tr_t$，把每個時間得到的reward合起來，這既是一個episode裡面，你得到的total reward。這個total reward才是我們需要去maximize的對象。我們不需要去maximize 每個step的reward，我們是要maximize 整個遊戲玩完之後的total reward。假設我們拿同一個Actor，每次玩的時候，$R_\theta$其實都會不一樣的。因為兩個原因，首先你Actor本身如果是Policy，看到同樣的場景它也會採取不同的Action。所以就算是同一個Actor，同一組參數，每次玩的時候你得到的$R_\theta$也會不一樣的。再來遊戲本身也有隨機性，就算你採取同一個Action，你看到的observation每次也可能都不一樣。所以$R_\theta$是一個Random Variable。我們做的事情，不是去maximize每次玩遊戲時的$R_\theta$，而是去maximize $R_\theta$的期望值。這個期望值就衡量了某一個Actor的好壞，好的Actor期望值就應該要比較大。

	那麼怎麼計算呢，我們假設一場遊戲就是一個trajectory $\tau$
	$$ \tau = \left\{ s_1,a_1,r_1, s_2,a_2,r_2,...,s_T,a_T,r_T \right\} $$
	$\tau$ 包含了state，看到這個observation，take的Action，得到的Reward，是一個sequence。
	$$
	R(\tau) = \sum_{n=1}^Nr_n
	$$
	$R(\tau)$代表在這個episode裡面，最後得到的總reward。當我們用某一個Actor去玩這個遊戲的時候，每個$\tau$都會有出現的幾率，$\tau$代表從遊戲開始到結束過程，這個過程有千百萬種，當你選擇這個Actor的時候，你可能只會看到某一些過程，某些過程特別容易出現，某些過程比較不容易出現。每個遊戲出現的過程，可以用一個幾率$P(\tau|\theta)$來表示它，就是說參數是$\theta$時$\tau$這個過程出現的幾率。那麼$R_\theta$的期望值為
	$$\bar{R}_\theta=\sum_\tau R(\tau)P(\tau|\theta)$$
	實際上要窮舉所有的$\tau$是不可能的，那麼要怎麼做？讓Actor去玩N場這個遊戲，獲得N個過程${\tau^1,\tau^2,...,\tau^N}$ ，玩N場就好像從$P(\tau|\theta)$去Sample N個$\tau$。假設某個$\tau$它的幾率特別大，就特別容易被sample出來。sample出來的$\tau$跟幾率成正比。讓Actor去玩N場，相當於從$P(\tau|\theta)$概率場抽取N個過程，可以通過N各Reward的均值進行近似，如下表達
	$$\bar{R}_\theta=\sum_\tau R(\tau)P(\tau|\theta) \approx \frac{1}{N}R(\tau^n)$$

- Pick the best function

	怎麼選擇最好的function，其實就是用我們的Gradient Ascent。我們已經找到目標了，就是最大化這個$\bar{R}_\theta$
	$$\theta^\ast = arg \max_\theta \bar{R}_\theta$$
	其中$\bar{R}_\theta = \sum_\tau R(\tau)P(\tau|\theta)$。就可以用Gradient Ascent進行最大化，過程為：
	(1) 初始化$\theta^0$ 
	(2) $\theta^1 \leftarrow \theta^0+\eta \triangledown \bar{R}_{\theta^0}$ 
	(3) $\theta^2 \leftarrow \theta^1+\eta \triangledown \bar{R}_{\theta^1}$ 
	(4) .......
	
	參數$\theta = {w_1,w_2,...,b_1,...}$，那麼$\triangledown \bar{R}_{\theta}$就是$\bar{R}_{\theta}$對每個參數的偏微分，如下
	$$
	\triangledown \bar{R}_{\theta} = \begin{bmatrix}
	\partial{ \bar{R}_{\theta}}/\partial w_1 \\ \partial{ \bar{R}_{\theta}}/\partial w_2
	\\ \vdots
	\\ \bar{R}_{\theta}/\partial b_1
	\\ \vdots
	\end{bmatrix} 
    $$接下來就是實際的計算下，$\bar{R}_\theta = \sum_\tau R(\tau)P(\tau|\theta)$中，只有$P(\tau|\theta)$跟$\theta$有關係，所以只需要對$P(\tau|\theta)$做Gradient ，即$$\nabla \bar{R}_{\theta}=\sum_{\tau} R(\tau) \nabla P(\tau | \theta)$$所以$R(\tau)$就算不可微也沒有關係，或者是不知道它的function也沒有差，我們只要知道把$\tau$放進去得到值就可以。
	接下來，為了讓$P(\tau|\theta)$出現，有$$\nabla \bar{R}_{\theta}=\sum_{\tau} R(\tau) \nabla P(\tau | \theta)=\sum_{\tau} R(\tau) P(\tau | \theta) \frac{\nabla P(\tau | \theta)}{P(\tau | \theta)}$$

	由於
	$$
	\frac{\operatorname{dlog}(f(x))}{d x}=\frac{1}{f(x)} \frac{d f(x)}{d x}
	$$
	所以
	$$
	\nabla \bar{R}_{\theta}=\sum_{\tau} R(\tau) P(\tau | \theta) \frac{\nabla P(\tau | \theta)}{P(\tau | \theta)}=\sum_{\tau} R(\tau) P(\tau | \theta) \nabla \log P(\tau | \theta)
	$$
	從而可以通過抽樣的方式去近似，即
	$$
	\nabla \bar{R}_{\theta}=\sum_{\tau} R(\tau) P(\tau | \theta) \nabla \log P(\tau | \theta)=\approx \frac{1}{N} \sum_{n=1}^{N} R\left(\tau^{n}\right) \nabla \log P\left(\tau^{n} | \theta\right)
	$$
	即拿$\theta$去玩N次遊戲，得到${\tau^1,\tau^2,...,\tau^N}$，算出每次的$R(\tau)$。接下來的問題是怎麼計算$\nabla \log P\left(\tau^{n} | \theta\right)$，因為
	$$
	P(\tau|\theta)=p\left(s_{1}\right) p\left(a_{1} | s_{1}, \theta\right) p\left(r_{1}, s_{2} | s_{1}, a_{1}\right) p\left(a_{2} | s_{2}, \theta\right) p\left(r_{2}, s_{3} | s_{2}, a_{2}\right) \cdots \\
	=p\left(s_{1}\right) \prod_{t=1}^{T} p\left(a_{t} | s_{t}, \theta\right) p\left(r_{t}, s_{t+1} | s_{t}, a_{t}\right)
	$$
	其中$P(s_1)$是初始狀態出現的幾率，接下來根據$\theta$會有某個概率在$s_1$狀態下採取Action $a_1$，然後根據$a_1,s_1$會得到某個reward $r_1$，並跳到另一個state $s_2$，以此類推。其中有些項跟Actor是無關的，$p\left(s_{1}\right)$和$p\left(r_{t}, s_{t+1} | s_{t}, a_{t}\right)$，只有$p\left(a_{t} | s_{t}, \theta\right)$跟Actor $\pi_\theta$有關係。
	通過取log，連乘轉為連加，即
	$$
	\log P(\tau | \theta) =\log p\left(s_{1}\right)+\sum_{t=1}^{T} \log p\left(a_{t} | s_{t}, \theta\right)+\log p\left(r_{t}, s_{t+1} | s_{t}, a_{t}\right) 
	$$
	然後對$\theta$取Gradient，刪除無關項，得到
	$$
	\nabla \log P(\tau | \theta)=\sum_{t=1}^{T} \nabla \log p\left(a_{t} | s_{t}, \theta\right)
	$$
	則
	$$
	\begin{aligned} \nabla \bar{R}_{\theta} & \approx \frac{1}{N} \sum_{n=1}^{N} R\left(\tau^{n}\right) \nabla \log P\left(\tau^{n} | \theta\right)=\frac{1}{N} \sum_{n=1}^{N} R\left(\tau^{n}\right) \sum_{t=1}^{T_{n}} \nabla \log p\left(a_{t}^{n} | s_{t}^{n}, \theta\right) \\ &=\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_{n}} R\left(\tau^{n}\right) \nabla \log p\left(a_{t}^{n} | s_{t}^{n}, \theta\right) \end{aligned}
	$$
	這個式子就告訴我們，當我們在某一次$\tau^n$遊戲中，在$s_t^n$狀態下採取$a_t^2$得到$R(\tau^n)$是正的，我們就希望$\theta$能夠使$p(a_t^n|s_t^n)$的概率越大越好。反之，如果$R(\tau^n)$是負的，就要調整$\theta$參數，能夠使$p(a_t^n|s_t^n)$的幾率變小。注意，某個時間點的$p\left(a_{t}^{n} | s_{t}^{n}, \theta\right)$是乘上這次遊戲的所有reward $R(\tau^n)$而不是這個時間點的reward。假設我們只考慮這個時間點的reward，那麼就是說只有fire才能得到reward，其他的action你得到的reward都是0。Machine就只會增加fire的幾率，不會增加left或者right的幾率。最後Learn出來的Agent它就會fire。

	接著還有一個問題，為什麼要取log呢？

	![39_12](./res/chapter39_12.png)

	$$\nabla \log p\left(a_{t}^{n} | s_{t}^{n}, \theta\right)=\frac{\nabla  p\left(a_{t}^{n} | s_{t}^{n}, \theta\right)}{p\left(a_{t}^{n} | s_{t}^{n}, \theta\right)}$$
	那麼為什麼要除以$p\left(a_{t}^{n} | s_{t}^{n}, \theta\right)$呢？假設某個state s在$\tau^{13},\tau^{15},\tau^{17},\tau^{33}$，採取不同的action，獲得不同的reward，會偏好出現次數比較多的action，但是次數比較多的action有時候並沒有比較好，就像b出現的比較多，但是得到的reward並沒有比較大，Machine把這項幾率調高。除掉一個幾率，就是對那項做了個normalization，防止Machine偏好那些出現幾率比較高的項。

	![39_13](./res/chapter39_13.png)

	還有另外一個問題，假設$R(\tau^n)$總是正的，那麼會出現什麼事情呢？在理想的狀態下，這件事情不會構成任何問題。假設有三個action，a,b,c採取的結果得到的reward都是正的，這個正有大有小，假設a和c的$R(\tau^n)$比較大，b的$R(\tau^n)$比較小，經過update之後，你還是會讓b出現的幾率變小，a,c出現的幾率變大，因為會做normalization。但是實做的時候，我們做的事情是sampling，所以有可能只sample b和c，這樣b,c幾率都會增加，a沒有sample到，幾率就自動減少，這樣就會有問題了。

	![39_14](./res/chapter39_14.png)

	這樣，我們就希望$R(\tau^n)$有正有負這樣，可以通過將$R(\tau^n)-b$來避免，$b$需要自己設計。如下
	$$\nabla \bar{R}_{\theta} \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_{n}}\left(R\left(\tau^{n}\right)-b\right) \nabla \log p\left(a_{t}^{n} | s_{t}^{n}, \theta\right)$$
	這樣$R(\tau^n)$超過b的時候就把幾率增加，小於b的時候就把幾率降低，從而解決了都是正的問題。

### Value-based 方法

#### Critic

Critic就是Learn一個Neural Network，這個Neural Network不做事，然後Actor可以從這個Critic中獲得，這就是Q-learning。
Critic就是learn一個function，這個function可以告訴你說現在看到某一個observation的時候，這個observation有有多好這樣。

- 根據actor $\pi$評估critic function

	這個function是用Neural Network表示 

- state value function $V^\pi(s)$

	這個累加的reward是通過觀察多個observation

	![39_15](./res/chapter39_15.png)

	那麼如何估計 $V^\pi(s)$呢？可以採用Monte-Carlo based approach。

- State-action value function $Q^\pi(s,a)$

	這個累加的reward是通過觀察observation和take的action

	![39_16](./res/chapter39_16.png)

### Actor-Critic

 這部分留著下學期再講

 ![39_17](./res/chapter39_17.png)

 附上一張學習地圖，完結撒花~

 ![39_18](./res/chapter39_18.png)
