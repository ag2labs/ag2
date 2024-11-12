"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[1759],{95427:(e,n,o)=>{o.r(n),o.d(n,{assets:()=>c,contentTitle:()=>d,default:()=>u,frontMatter:()=>r,metadata:()=>s,toc:()=>a});var t=o(85893),i=o(11151);const r={},d="Docker for Development",s={id:"contributor-guide/docker",title:"Docker for Development",description:"For developers contributing to the AutoGen project, we offer a specialized Docker environment. This setup is designed to streamline the development process, ensuring that all contributors work within a consistent and well-equipped environment.",source:"@site/docs/contributor-guide/docker.md",sourceDirName:"contributor-guide",slug:"/contributor-guide/docker",permalink:"/autogen/docs/contributor-guide/docker",draft:!1,unlisted:!1,editUrl:"https://github.com/ag2ai/ag2/edit/main/website/docs/contributor-guide/docker.md",tags:[],version:"current",frontMatter:{},sidebar:"docsSidebar",previous:{title:"Contributing to AutoGen",permalink:"/autogen/docs/contributor-guide/contributing"},next:{title:"Documentation",permalink:"/autogen/docs/contributor-guide/documentation"}},c={},a=[{value:"Autogen Developer Image (ag2_dev_img)",id:"autogen-developer-image-ag2_dev_img",level:2},{value:"Building the Developer Docker Image",id:"building-the-developer-docker-image",level:2},{value:"Using the Developer Docker Image",id:"using-the-developer-docker-image",level:2},{value:"Develop in Remote Container",id:"develop-in-remote-container",level:2}];function l(e){const n={a:"a",code:"code",h1:"h1",h2:"h2",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,i.a)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(n.h1,{id:"docker-for-development",children:"Docker for Development"}),"\n",(0,t.jsx)(n.p,{children:"For developers contributing to the AutoGen project, we offer a specialized Docker environment. This setup is designed to streamline the development process, ensuring that all contributors work within a consistent and well-equipped environment."}),"\n",(0,t.jsx)(n.h2,{id:"autogen-developer-image-ag2_dev_img",children:"Autogen Developer Image (ag2_dev_img)"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Purpose"}),": The ",(0,t.jsx)(n.code,{children:"ag2_dev_img"})," is tailored for contributors to the AutoGen project. It includes a suite of tools and configurations that aid in the development and testing of new features or fixes."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Usage"}),": This image is recommended for developers who intend to contribute code or documentation to AutoGen."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Forking the Project"}),": It's advisable to fork the AutoGen GitHub project to your own repository. This allows you to make changes in a separate environment without affecting the main project."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Updating Dockerfile"}),": Modify your copy of ",(0,t.jsx)(n.code,{children:"Dockerfile"})," in the ",(0,t.jsx)(n.code,{children:"dev"})," folder as needed for your development work."]}),"\n",(0,t.jsxs)(n.li,{children:[(0,t.jsx)(n.strong,{children:"Submitting Pull Requests"}),": Once your changes are ready, submit a pull request from your branch to the upstream AutoGen GitHub project for review and integration. For more details on contributing, see the ",(0,t.jsx)(n.a,{href:"https://ag2ai.github.io/autogen/docs/Contribute",children:"AutoGen Contributing"})," page."]}),"\n"]}),"\n",(0,t.jsx)(n.h2,{id:"building-the-developer-docker-image",children:"Building the Developer Docker Image"}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:["\n",(0,t.jsxs)(n.p,{children:["To build the developer Docker image (",(0,t.jsx)(n.code,{children:"ag2_dev_img"}),"), use the following commands:"]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-bash",children:"docker build -f .devcontainer/dev/Dockerfile -t ag2_dev_img https://github.com/ag2ai/ag2.git#main\n"})}),"\n"]}),"\n",(0,t.jsxs)(n.li,{children:["\n",(0,t.jsx)(n.p,{children:"For building the developer image built from a specific Dockerfile in a branch other than main/master"}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-bash",children:"# clone the branch you want to work out of\ngit clone --branch {branch-name} https://github.com/ag2ai/ag2.git\n\n# cd to your new directory\ncd autogen\n\n# build your Docker image\ndocker build -f .devcontainer/dev/Dockerfile -t autogen_dev-srv_img .\n"})}),"\n"]}),"\n"]}),"\n",(0,t.jsx)(n.h2,{id:"using-the-developer-docker-image",children:"Using the Developer Docker Image"}),"\n",(0,t.jsxs)(n.p,{children:["Once you have built the ",(0,t.jsx)(n.code,{children:"ag2_dev_img"}),", you can run it using the standard Docker commands. This will place you inside the containerized development environment where you can run tests, develop code, and ensure everything is functioning as expected before submitting your contributions."]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-bash",children:"docker run -it -p 8081:3000 -v `pwd`/autogen-newcode:newstuff/ ag2_dev_img bash\n"})}),"\n",(0,t.jsxs)(n.ul,{children:["\n",(0,t.jsxs)(n.li,{children:["Note that the ",(0,t.jsx)(n.code,{children:"pwd"}),' is shorthand for present working directory. Thus, any path after the pwd is relative to that. If you want a more verbose method you could remove the "',(0,t.jsx)(n.code,{children:"pwd"}),'/autogen-newcode" and replace it with the full path to your directory']}),"\n"]}),"\n",(0,t.jsx)(n.pre,{children:(0,t.jsx)(n.code,{className:"language-bash",children:"docker run -it -p 8081:3000 -v /home/AutoGenDeveloper/autogen-newcode:newstuff/ ag2_dev_img bash\n"})}),"\n",(0,t.jsx)(n.h2,{id:"develop-in-remote-container",children:"Develop in Remote Container"}),"\n",(0,t.jsxs)(n.p,{children:["If you use vscode, you can open the autogen folder in a ",(0,t.jsx)(n.a,{href:"https://code.visualstudio.com/docs/remote/containers",children:"Container"}),".\nWe have provided the configuration in ",(0,t.jsx)(n.a,{href:"https://github.com/ag2ai/ag2/blob/main/.devcontainer",children:"devcontainer"}),". They can be used in GitHub codespace too. Developing AutoGen in dev containers is recommended."]})]})}function u(e={}){const{wrapper:n}={...(0,i.a)(),...e.components};return n?(0,t.jsx)(n,{...e,children:(0,t.jsx)(l,{...e})}):l(e)}},11151:(e,n,o)=>{o.d(n,{Z:()=>s,a:()=>d});var t=o(67294);const i={},r=t.createContext(i);function d(e){const n=t.useContext(r);return t.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function s(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:d(e.components),t.createElement(r.Provider,{value:n},e.children)}}}]);