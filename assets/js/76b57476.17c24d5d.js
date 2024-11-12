"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[3727],{86529:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>c,contentTitle:()=>i,default:()=>h,frontMatter:()=>s,metadata:()=>r,toc:()=>l});var o=t(85893),a=t(11151);const s={custom_edit_url:"https://github.com/ag2ai/ag2/edit/main/website/docs/tutorial/introduction.ipynb",source_notebook:"/website/docs/tutorial/introduction.ipynb",title:"Introduction to AutoGen"},i="Introduction to AutoGen",r={id:"tutorial/introduction",title:"Introduction to AutoGen",description:"Open In Colab",source:"@site/docs/tutorial/introduction.mdx",sourceDirName:"tutorial",slug:"/tutorial/introduction",permalink:"/autogen/docs/tutorial/introduction",draft:!1,unlisted:!1,editUrl:"https://github.com/ag2ai/ag2/edit/main/website/docs/tutorial/introduction.ipynb",tags:[],version:"current",frontMatter:{custom_edit_url:"https://github.com/ag2ai/ag2/edit/main/website/docs/tutorial/introduction.ipynb",source_notebook:"/website/docs/tutorial/introduction.ipynb",title:"Introduction to AutoGen"},sidebar:"docsSidebar",previous:{title:"Tutorial",permalink:"/autogen/docs/tutorial"},next:{title:"Terminating Conversations Between Agents",permalink:"/autogen/docs/tutorial/chat-termination"}},c={},l=[{value:"Why AutoGen?",id:"why-autogen",level:2},{value:"Installation",id:"installation",level:2},{value:"Agents",id:"agents",level:2},{value:"Roles and Conversations",id:"roles-and-conversations",level:2},{value:"Summary",id:"summary",level:2}];function d(e){const n={a:"a",blockquote:"blockquote",code:"code",em:"em",h1:"h1",h2:"h2",img:"img",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",...(0,a.a)(),...e.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(n.h1,{id:"introduction-to-autogen",children:"Introduction to AutoGen"}),"\n",(0,o.jsxs)(n.p,{children:[(0,o.jsx)(n.a,{href:"https://colab.research.google.com/github/ag2ai/ag2/blob/main/website/docs/tutorial/introduction.ipynb",children:(0,o.jsx)(n.img,{src:"https://colab.research.google.com/assets/colab-badge.svg",alt:"Open In Colab"})}),"\n",(0,o.jsx)(n.a,{href:"https://github.com/ag2ai/ag2/blob/main/website/docs/tutorial/introduction.ipynb",children:(0,o.jsx)(n.img,{src:"https://img.shields.io/badge/Open%20on%20GitHub-grey?logo=github",alt:"Open on GitHub"})})]}),"\n",(0,o.jsxs)(n.p,{children:["Welcome! AutoGen is an open-source framework that leverages multiple\n",(0,o.jsx)(n.em,{children:"agents"})," to enable complex workflows. This tutorial introduces basic\nconcepts and building blocks of AutoGen."]}),"\n",(0,o.jsx)(n.h2,{id:"why-autogen",children:"Why AutoGen?"}),"\n",(0,o.jsxs)(n.blockquote,{children:["\n",(0,o.jsxs)(n.p,{children:[(0,o.jsx)(n.em,{children:"The whole is greater than the sum of its parts."}),(0,o.jsx)("br",{})," -",(0,o.jsx)(n.strong,{children:"Aristotle"})]}),"\n"]}),"\n",(0,o.jsx)(n.p,{children:"While there are many definitions of agents, in AutoGen, an agent is an\nentity that can send messages, receive messages and generate a reply\nusing models, tools, human inputs or a mixture of them. This abstraction\nnot only allows agents to model real-world and abstract entities, such\nas people and algorithms, but it also simplifies implementation of\ncomplex workflows as collaboration among agents."}),"\n",(0,o.jsx)(n.p,{children:"Further, AutoGen is extensible and composable: you can extend a simple\nagent with customizable components and create workflows that can combine\nthese agents and power a more sophisticated agent, resulting in\nimplementations that are modular and easy to maintain."}),"\n",(0,o.jsx)(n.p,{children:"Most importantly, AutoGen is developed by a vibrant community of\nresearchers and engineers. It incorporates the latest research in\nmulti-agent systems and has been used in many real-world applications,\nincluding agent platform, advertising, AI employees, blog/article\nwriting, blockchain, calculate burned areas by wildfires, customer\nsupport, cybersecurity, data analytics, debate, education, finance,\ngaming, legal consultation, research, robotics, sales/marketing, social\nsimulation, software engineering, software security, supply chain,\nt-shirt design,\xa0training data generation, Youtube\xa0service\u2026"}),"\n",(0,o.jsx)(n.h2,{id:"installation",children:"Installation"}),"\n",(0,o.jsxs)(n.p,{children:["The simplest way to install AutoGen is from pip: ",(0,o.jsx)(n.code,{children:"pip install autogen"}),".\nFind more options in ",(0,o.jsx)(n.a,{href:"../../docs/installation/",children:"Installation"}),"."]}),"\n",(0,o.jsx)(n.h2,{id:"agents",children:"Agents"}),"\n",(0,o.jsx)(n.p,{children:"In AutoGen, an agent is an entity that can send and receive messages to\nand from other agents in its environment. An agent can be powered by\nmodels (such as a large language model like GPT-4), code executors (such\nas an IPython kernel), human, or a combination of these and other\npluggable and customizable components."}),"\n",(0,o.jsx)(n.p,{children:(0,o.jsx)(n.img,{alt:"ConversableAgent",src:t(65575).Z+"",width:"579",height:"410"})}),"\n",(0,o.jsxs)(n.p,{children:["An example of such agents is the built-in ",(0,o.jsx)(n.code,{children:"ConversableAgent"})," which\nsupports the following components:"]}),"\n",(0,o.jsxs)(n.ol,{children:["\n",(0,o.jsx)(n.li,{children:"A list of LLMs"}),"\n",(0,o.jsx)(n.li,{children:"A code executor"}),"\n",(0,o.jsx)(n.li,{children:"A function and tool executor"}),"\n",(0,o.jsx)(n.li,{children:"A component for keeping human-in-the-loop"}),"\n"]}),"\n",(0,o.jsxs)(n.p,{children:["You can switch each component on or off and customize it to suit the\nneed of your application. For advanced users, you can add additional\ncomponents to the agent by using\n",(0,o.jsx)(n.a,{href:"../reference/agentchat/conversable_agent/#register_reply",children:(0,o.jsx)(n.code,{children:"registered_reply"})}),"."]}),"\n",(0,o.jsxs)(n.p,{children:["LLMs, for example, enable agents to converse in natural languages and\ntransform between structured and unstructured text. The following\nexample shows a ",(0,o.jsx)(n.code,{children:"ConversableAgent"})," with a GPT-4 LLM switched on and\nother components switched off:"]}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-python",children:'import os\n\nfrom autogen import ConversableAgent\n\nagent = ConversableAgent(\n    "chatbot",\n    llm_config={"config_list": [{"model": "gpt-4", "api_key": os.environ.get("OPENAI_API_KEY")}]},\n    code_execution_config=False,  # Turn off code execution, by default it is off.\n    function_map=None,  # No registered functions, by default it is None.\n    human_input_mode="NEVER",  # Never ask for human input.\n)\n'})}),"\n",(0,o.jsxs)(n.p,{children:["The ",(0,o.jsx)(n.code,{children:"llm_config"})," argument contains a list of configurations for the\nLLMs. See ",(0,o.jsx)(n.a,{href:"../../docs/topics/llm_configuration",children:"LLM Configuration"})," for\nmore details."]}),"\n",(0,o.jsxs)(n.p,{children:["You can ask this agent to generate a response to a question using the\n",(0,o.jsx)(n.code,{children:"generate_reply"})," method:"]}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-python",children:'reply = agent.generate_reply(messages=[{"content": "Tell me a joke.", "role": "user"}])\nprint(reply)\n'})}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-text",children:"Sure, here's a light-hearted joke for you:\n\nWhy don't scientists trust atoms?\n\nBecause they make up everything!\n"})}),"\n",(0,o.jsx)(n.h2,{id:"roles-and-conversations",children:"Roles and Conversations"}),"\n",(0,o.jsxs)(n.p,{children:["In AutoGen, you can assign roles to agents and have them participate in\nconversations or chat with each other. A conversation is a sequence of\nmessages exchanged between agents. You can then use these conversations\nto make progress on a task. For example, in the example below, we assign\ndifferent roles to two agents by setting their ",(0,o.jsx)(n.code,{children:"system_message"}),"."]}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-python",children:'cathy = ConversableAgent(\n    "cathy",\n    system_message="Your name is Cathy and you are a part of a duo of comedians.",\n    llm_config={"config_list": [{"model": "gpt-4", "temperature": 0.9, "api_key": os.environ.get("OPENAI_API_KEY")}]},\n    human_input_mode="NEVER",  # Never ask for human input.\n)\n\njoe = ConversableAgent(\n    "joe",\n    system_message="Your name is Joe and you are a part of a duo of comedians.",\n    llm_config={"config_list": [{"model": "gpt-4", "temperature": 0.7, "api_key": os.environ.get("OPENAI_API_KEY")}]},\n    human_input_mode="NEVER",  # Never ask for human input.\n)\n'})}),"\n",(0,o.jsxs)(n.p,{children:["Now that we have two comedian agents, we can ask them to start a comedy\nshow. This can be done using the ",(0,o.jsx)(n.code,{children:"initiate_chat"})," method. We set the\n",(0,o.jsx)(n.code,{children:"max_turns"})," to 2 to keep the conversation short."]}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-python",children:'result = joe.initiate_chat(cathy, message="Cathy, tell me a joke.", max_turns=2)\n'})}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-text",children:"joe (to cathy):\n\nCathy, tell me a joke.\n\n--------------------------------------------------------------------------------\ncathy (to joe):\n\nSure, here's one for you:\n\nWhy don't scientists trust atoms?\n\nBecause they make up everything!\n\n--------------------------------------------------------------------------------\njoe (to cathy):\n\nHaha, that's a good one, Cathy! Okay, my turn. \n\nWhy don't we ever tell secrets on a farm?\n\nBecause the potatoes have eyes, the corn has ears, and the beans stalk.\n\n--------------------------------------------------------------------------------\ncathy (to joe):\n\nHaha, that's a great one! A farm is definitely not the place for secrets. Okay, my turn again. \n\nWhy couldn't the bicycle stand up by itself?\n\nBecause it was two-tired!\n\n--------------------------------------------------------------------------------\n"})}),"\n",(0,o.jsx)(n.p,{children:"The comedians are bouncing off each other!"}),"\n",(0,o.jsx)(n.h2,{id:"summary",children:"Summary"}),"\n",(0,o.jsxs)(n.p,{children:["In this chapter, we introduced the concept of agents, roles and\nconversations in AutoGen. For simplicity, we only used LLMs and created\nfully autonomous agents (",(0,o.jsx)(n.code,{children:"human_input_mode"})," was set to ",(0,o.jsx)(n.code,{children:"NEVER"}),"). In the\nnext chapter, we will show how you can control when to ",(0,o.jsx)(n.em,{children:"terminate"})," a\nconversation between autonomous agents."]})]})}function h(e={}){const{wrapper:n}={...(0,a.a)(),...e.components};return n?(0,o.jsx)(n,{...e,children:(0,o.jsx)(d,{...e})}):d(e)}},65575:(e,n,t)=>{t.d(n,{Z:()=>o});const o=t.p+"assets/images/conversable-agent-58656a6f9b08b1d7d942a358e22a28bb.jpg"},11151:(e,n,t)=>{t.d(n,{Z:()=>r,a:()=>i});var o=t(67294);const a={},s=o.createContext(a);function i(e){const n=o.useContext(s);return o.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function r(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(a):e.components||a:i(e.components),o.createElement(s.Provider,{value:n},e.children)}}}]);