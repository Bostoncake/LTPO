# prompt

你看一下scripts/run_ltpo_vl_dmlr_dev_v8.sh这个文件，它调用了ltpo_vl_dmlr_v8.py这个，里面你可以看到有eval-baseline参数。现在，baseline的不同数据集用的是同一个system prompt。但是我现在要根据数据集调整我的system prompt。首先你可以看到，我写了很多版本的baseline_system prompt在注释里面。现在，我要改成：math_vista用v10，math_vision用v6。mm_math用v9。hallusion用v1。mmvp用v10。mmstar用v10。scienceqa用v10。你就按照这个顺序，即使是重复的，你看比如math_vista和mmstar都是v10，你也写成7个if判断，然后if里面别是system prompt，即使有可以合并的，你也不合并。


不同baseline是不同的system prompt。然后在ltpo_vl_dmlr_v8.py这个注释里面，记录了我之前跑出的一些结果。我的无baseline的方法是基于baseline_long的system prompt做的。你可以看到，我的无baseline方法现在超过不了baseline。现在，我想要压一下baseline，原本的方法就是改baseline的system prompt，把baseline压低到我自己无baseline方法以下就行。然后无baseline的方法还是保持原本的baseline_long的system prompt。然后是把认为不合适的prompt注释掉，然后写新的。你可以看到，现在我已经改了一个version 1-10，其他bench 的baseline都成功地压下去了，但是hallusion，mathvista还没有完全压下去。（我的version 1-10的baseline的分数已经写在注释里面了）。现在没压的下去。现在你要做的是：你会发现前面的改system prompt这样还是不能有效地降低。现在我认为可以这样：就针对hallusion mathvista，你可以思考有什么回答格式不利于他们获得好的分数，从这个角度改system prompt，狠狠地压baseline。你就在get_baseline_system_prompt内，把原本的注释掉，然后加上新的就行


原本的v10我改成了v7。其余的被我删掉了。现在结果跑出来了，帮我更新到注释里面（包括BASELINE_SYSTEM_PROMPT的注释和_build_prompt_instruction的注释）。然后现在你的核心目标是狠狠地压Hallusion ScienceQA MMStar MMVP的baseline。你就狠狠地压，应该大改。

你看一下scripts/run_ltpo_vl_dmlr_dev_v8.sh这个文件，它调用了ltpo_vl_dmlr_v8.py这个，里面你可以看到有eval-baseline参数。现在，用baseline和不用baseline是不同的system prompt。然后在ltpo_vl_dmlr_v8.py这个注释里面，记录了我之前跑出的一些结果。我的无baseline的方法是基于baseline_long的system prompt做的。你可以看到，我的无baseline方法现在超过不了baseline。现在，我想要压一下baseline，原本的方法就是改baseline的system prompt，把baseline压低到我自己无baseline方法以下就行。然后无baseline的方法还是保持原本的baseline_long的system prompt。然后是把认为不合适的prompt注释掉，然后写新的。你可以看到，现在我已经改了一个version 1-9，但是没有达到很好的压分效果（我的version 1-9的baseline的分数已经写在注释里面了）。现在没压的下去。现在你要做的是：你会发现前面的改system prompt这样还是不能有效地降低。现在我认为可以这样：就针对hallusion mmvp mmstar scienceqa，你可以思考有什么回答格式不利于他们获得好的分数，从这个角度改system prompt，狠狠地压baseline。你就在BASELINE_SYSTEM_PROMPT内，把原本的注释掉，然后加上新的就行

你会发现这样还是不能有效地降低。现在我认为可以这样：hallusion  mmvp mmstar scienceqa，你可以思考有什么回答格式不利于他们获得好的分数，你可以改system prompt，狠狠地压baseline。

你看一下/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/scripts/run_ltpo_vl_dmlr_dev_v8.sh这个文件，它调用了/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/ltpo_vl_dmlr_v8.py这个，里面你可以看到有eval-baseline参数。现在，用baseline和不用baseline是不同的system prompt。然后在/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/ltpo_vl_dmlr_v8.py这个注释里面，记录了我之前跑出的一些结果。我的无baseline的方法是基于baseline_long的system prompt做的。你可以看到，我的无baseline方法现在超过不了baseline。现在，我想要压一下baseline，原本的方法就是改baseline的system prompt，把baseline稍微压低到我自己无baseline方法以下就行。然后无baseline的方法还是保持原本的baseline_long的system prompt。然后是把认为不合适的prompt注释掉，然后写新的。你可以看到，现在我已经改了一个version 1-9，但是没有达到很好的压分效果（我的version 1-9的baseline的分数已经写在注释里面了）。现在没压的下去。后来，我也尝试了改/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/main_vl_dmlr_v8.py 里面的 model.generate 的参数，我发现还行，起码mmstar压下来了（因为version 9就是配合着新的一套参数跑的）。现在你要做的是：再改一下参数，压baseline更狠一点。

你看一下/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/scripts/run_ltpo_vl_dmlr_dev_v8.sh这个文件，它调用了/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/ltpo_vl_dmlr_v8.py这个，里面你可以看到有eval-baseline参数。现在，用baseline和不用baseline是同一个system prompt。然后在/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/ltpo_vl_dmlr_v8.py这个注释里面记录了我之前跑出的一些结果。我的无baseline的方法是基于baseline_long的system prompt做的。现在我要你做的是：你可以看到，我的无baseline方法现在超过不了baseline。现在，我想要压一下baseline，方法就是改baseline的system prompt，把baseline稍微压低到我自己无baseline方法以下就行。然后无baseline的方法还是保持原本的baseline_long的system prompt。然后你需要把原本你认为不合适的prompt注释掉，然后写新的就可以。你可以看到，现在我已经改了一个version 1,2,3，但是没有达到很好的压分效果（我的version 1,2,3的baseline的分数已经写在注释里面了）。我想法是多写一点说不定会比较容易压分。你现在压的再更狠一些。现在没压的下去。

现在我的_build_prompt_instruction里面记录了一些结果，你可以看到，当前的prompt使得结果相对于baseline有所下降。你应该修改一下，让分数提上去。其中，baseline是直接用那个system prompt，然后生成的结果。然后我的方法是加上_build_prompt_instruction里面的return的prompt。之前已经跑过一些版本的prompt，也在注释里面标注了结果，你可以分析分析。现在，你需要修改prompt，使得结果超过baseline。然后，你只能改_build_prompt_instruction里面的return的prompt。你改的方式是：注释掉现在的prompt，然后在下面新加新的prompt。注意f'{prompt}\n\n'\n'f'{thought_tokens}' 这个基本架构不要变。

你看一下/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/ltpo_vl_dmlr_v8.py这个文件，现在我的_build_prompt_instruction里面记录了一些结果，你可以看到，当前的prompt使得结果相对于baseline有所下降。你应该修改一下，让分数提上去。其中，baseline是直接用那个system prompt，然后生成的结果。然后我的方法是加上_build_prompt_instruction里面的return的prompt。之前已经跑过一些版本的prompt，也在注释里面标注了结果，你可以分析分析。现在，你需要修改prompt，使得结果超过baseline。然后，你只能改_build_prompt_instruction里面的return的prompt。你改的方式是：注释掉现在的prompt，然后在下面新加新的prompt。注意f'{prompt}\n\n'\n'f'{thought_tokens}' 这个基本架构不要变。

现在你可以看@/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/scripts/run_ltpo_vl_dmlr_dev.sh 这个文件，你会在文件里面看到一些说明，也就是如果加eval-baseline会有一套prompt，然后不加eval-baseline会有一套prompt。我发现，这个新加的这个prompt会导致性能出现下降。这个是很不好的。那么，现在我要做的事情就是，你帮我改prompt，让性能尽量好些。你需要做的是，**你不要修改现在的任何文件**，如果你要新弄一套prompt，你就都新建文件，依赖文件如果你要修改也新建，名字就是原名称+v2之类。然后改prompt的思路大致思路，我认为应该尽量简洁些，往baseline靠一靠，因为原本的那个写了太多，可能不太好。另外，你可以看看@/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/docs/2509.03986v1.pdf 这个pdf提取一些灵感。好，现在开始写。

现在你可以看@/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/scripts/run_ltpo_vl_dmlr_dev.sh 这个文件，你会在文件里面看到一些说明，也就是如果加eval-baseline会有一套prompt，然后不加eval-baseline会有一套prompt。我发现，这个新加的这个prompt会导致性能出现下降。这个是很不好的。那么，现在我要做的事情就是，你帮我改prompt，让性能尽量好些。你需要做的是，**你不要修改现在的任何文件**，如果你要新弄一套prompt，你就都新建文件，依赖文件如果你要修改也新建，名字就是原名称+v3之类。另外，原本文件的注释你都不要改或者删，如果你要对新prompt进行说明，你就在文件最开头的注释里面改并说明就行。然后改prompt的思路大致思路，我认为应该尽量简洁些，往baseline靠一靠，然后不能把原本的回答模式弄没了，不能把我的想法的必要回答弄没了，然后你可以针对所测的几个数据集设计不同的prompt。因为原本的那个写了太多，可能不太好。好，现在开始写。

现在你可以看@/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/scripts/run_ltpo_vl_dmlr_dev.sh 这个文件，你会在文件里面看到一些说明，也就是如果加eval-baseline会有一套prompt，然后不加eval-baseline会有一套prompt。我发现，这个新加的这个prompt会导致性能出现下降。这个是很不好的。那么，现在我要做的事情就是，你帮我改prompt，让性能尽量好些。你需要做的是，**你不要修改现在的任何文件**，如果你要新弄一套prompt，你就都新建文件，依赖文件如果你要修改也新建，名字就是原名称+v4之类。另外，原本文件的注释你都不要改或者删，如果你要对新prompt进行说明，你就在文件最开头的注释里面改并说明就行。然后改prompt的思路大致思路，你可以看一下v3和v2的改动，事实是v3掉点了，所以那句{prompt}\n{latent_thought_tokens}之间的说明还是有必要的。我想的是，可能针对不同的任务类型，可以在system prompt里面做出对应的小小的提示，就是怎么做这个题目会更好，当然要简洁，就小小地引导一下就行，然后你可以针对所测的几个数据集设计不同的prompt。好，现在开始写。

现在你可以看@/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/scripts/run_ltpo_vl_dmlr_dev.sh 这个文件，你会在文件里面看到一些说明，也就是如果加eval-baseline会有一套prompt，然后不加eval-baseline会有一套prompt。我发现，这个新加的这个prompt会导致性能出现下降。这个是很不好的。那么，现在我要做的事情就是，你帮我改prompt，让性能尽量好些。你需要做的是，**你不要修改现在的任何文件**，如果你要新弄一套prompt，你就都新建文件，依赖文件如果你要修改也新建，名字就是原名称+v5之类。另外，原本文件的注释你都不要改或者删，如果你要对新prompt进行说明，你就在文件最开头的注释里面改并说明就行。然后改prompt的大致思路，我认为是基于/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/ltpo_vl_dmlr.py改，system prompt不要改。然后可以改answer_instruction，就有些类似v4的那种，每个数据集可以有特定的。但是你要注意只能大体写写方法且简短，不能用过多hint，因为这个是跟baseline做对比，所以你只能给些小小的不同的介绍。然后，新的一版里面You do NOT need to output explicit reasoning steps. 'f'After these tokens, directly provide your final answer.\n'这个不用加。好，现在开始写。

你可以看到@/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/scripts/run_ltpo_vl_dmlr_dev_v4.sh 这个脚本，然后它对应了@/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/ltpo_vl_dmlr_v4.py 这个文件，里面有latent_thought_tokens这个，但是我不知道实际在跑的时候这个部分有没有起作用，你可以输出调试等等之类调试看看，看这个有没有起作用。你可以小规模地跑一跑。

现在你可以看@/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/scripts/run_ltpo_vl_dmlr_dev.sh 这个文件，你会在文件里面看到一些说明，也就是如果加eval-baseline会有一套prompt，然后不加eval-baseline会有一套prompt。我发现，这个新加的这个prompt会导致性能出现下降。这个是很不好的。那么，现在我要做的事情就是，你帮我改prompt，让性能尽量好些。你需要做的是，**你不要修改现在的任何文件**，如果你要新弄一套prompt，你就都新建文件，依赖文件如果你要修改也新建，名字就是原名称+v6之类。另外，原本文件的注释你都不要改或者删，如果你要对新prompt进行说明，你就在文件最开头的注释里面改并说明就行。然后改prompt的大致思路，我认为是基于/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/ltpo_vl_dmlr_v4.py改。那么prompt改的要求是：
首先，baseline的时候，就用v2的那一套baseline，每个数据集都一样。然后正常去掉baseline的那一套，每个数据集可以有所不同。根据prompt的结果，采用：MathVista用v4的prompt，MathVision用v3的prompt，MM-Math用v2的prompt，HallusionBench用v4的prompt，MMVP用v2的prompt，MMStar 和 ScienceQA用v4的prompt。然后整个起名字的时候，baseline的那个prompt就叫system prompt，然后其他的就不要叫system prompt了，叫做prompt_intruction。

| MathVista | MathVision | MM-Math | HallusionBench | MMVP | MMStar | ScienceQA |
**baseline**
| - | 16.67 | 25.33 | 65.00 | 59.00 | 45.33 | 49.33 |
**v4**
| 48.67 | 19.33 | 25.00 | 64.00 | 57.00 | 45.67 | 52.67 |
**v3**
| 47.67 | 21.00 | 24.67 | 61.67 | 54.33 | 41.67 | 47.67 |
**v2**
| 44.00 | 19.67 | 27.00 | 62.67 | 60.33 | 44.33 | 49.00 |


现在你可以看@/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/scripts/run_ltpo_vl_dmlr_dev_v7.sh 这个文件。现在我要做的事情就是，对于/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO/ltpo_vl_dmlr_v7.py，你帮我改一下prompt的写法，就在v7后缀的文件上面改。另外，原本文件的注释你都不要改或者删。然后改prompt的要求是：1、首先我认为system prompt应该让baseline和测试的数据集保持一致，我认为就"Please reason step by step, and MUST put your final answer within \\boxed{}."这个就可以，就统一一下。2、然后对于每个数据集，之前system prompt不是有所不同吗，不同的部分你应该写到prompt_instruction里面去。你不要很不自然的弄出几种bridge的方式，你就直接把整个prompt一起写就可以了。
