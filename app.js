(function(){
'use strict';
const W=640,H=360,HOR=150,VPX=320,K=280;
const DET_R=10.5,PASS_D=7,NCONF=8,PTHR=0.7,SCALE=180,MASKR=14.5,DET_D=Math.round(K/DET_R);

const CLASSES=[
  {name:'Speed limit 30',kind:'speed',val:30},
  {name:'Speed limit 50',kind:'speed',val:50},
  {name:'Speed limit 60',kind:'speed',val:60},
  {name:'Speed limit 80',kind:'speed',val:80},
  {name:'Speed limit 100',kind:'speed',val:100},
  {name:'Speed limit 120',kind:'speed',val:120},
  {name:'Stop',kind:'stop'},
  {name:'Yield',kind:'yield'},
  {name:'No entry',kind:'noentry'},
  {name:'Priority road',kind:'priority'},
  {name:'Ahead only',kind:'ahead'},
  {name:'Turn right ahead',kind:'right'},
  {name:'General caution',kind:'caution'},
  {name:'Road work',kind:'works'}
];

/* ---------- sign drawing (used for both scene and templates) ---------- */
const RED='#d12b2b',BLUE='#1461b5',WHITE='#f7f7f3',YEL='#f3c614',BLK='#171717';
function poly(c,pts,fill){c.beginPath();for(let i=0;i<pts.length;i++){if(i)c.lineTo(pts[i][0],pts[i][1]);else c.moveTo(pts[i][0],pts[i][1]);}c.closePath();c.fillStyle=fill;c.fill();}
function disc(c,x,y,r,fill){c.beginPath();c.arc(x,y,r,0,Math.PI*2);c.fillStyle=fill;c.fill();}
function ngon(x,y,r,n,rot){const a=[];for(let i=0;i<n;i++){const t=rot+i*2*Math.PI/n;a.push([x+r*Math.cos(t),y+r*Math.sin(t)]);}return a;}
function triUp(x,y,r,s){s=s||1;return [[x,y-r*s],[x-r*.866*s,y+r*.5*s],[x+r*.866*s,y+r*.5*s]];}
function triDown(x,y,r,s){s=s||1;return [[x-r*.866*s,y-r*.5*s],[x+r*.866*s,y-r*.5*s],[x,y+r*s]];}
function dia(x,y,r){return [[x,y-r],[x+r,y],[x,y+r],[x-r,y]];}
function txt(c,t,x,y,size,fill){c.font='700 '+size+'px Arial,Helvetica,sans-serif';c.textAlign='center';c.textBaseline='middle';c.fillStyle=fill;c.fillText(t,x,y);}
function drawPole(c,x,y,r,len){c.fillStyle='#8a8e93';const w=Math.max(1,r*.13);c.fillRect(x-w/2,y,w,len);}
function drawSign(c,k,x,y,r){
  switch(k.kind){
    case 'speed':{disc(c,x,y,r,RED);disc(c,x,y,r*.79,WHITE);const s=String(k.val);txt(c,s,x,y+r*.03,r*(s.length>2?.6:.8),BLK);break;}
    case 'stop':{const rot=Math.PI/8;poly(c,ngon(x,y,r,8,rot),RED);poly(c,ngon(x,y,r*.9,8,rot),WHITE);poly(c,ngon(x,y,r*.82,8,rot),RED);txt(c,'STOP',x,y+r*.02,r*.5,WHITE);break;}
    case 'yield':{poly(c,triDown(x,y,r),RED);poly(c,triDown(x,y,r,.62),WHITE);break;}
    case 'noentry':{disc(c,x,y,r,RED);c.fillStyle=WHITE;c.fillRect(x-r*.7,y-r*.16,r*1.4,r*.32);break;}
    case 'priority':{poly(c,dia(x,y,r),BLK);poly(c,dia(x,y,r*.93),WHITE);poly(c,dia(x,y,r*.72),YEL);break;}
    case 'ahead':{disc(c,x,y,r,BLUE);c.fillStyle=WHITE;c.fillRect(x-r*.11,y-r*.15,r*.22,r*.7);poly(c,[[x,y-r*.65],[x-r*.36,y-r*.15],[x+r*.36,y-r*.15]],WHITE);break;}
    case 'right':{disc(c,x,y,r,BLUE);c.fillStyle=WHITE;c.fillRect(x-r*.31,y-r*.2,r*.22,r*.8);c.fillRect(x-r*.31,y-r*.2,r*.6,r*.22);poly(c,[[x+r*.64,y-r*.09],[x+r*.26,y-r*.45],[x+r*.26,y+r*.27]],WHITE);break;}
    case 'caution':{poly(c,triUp(x,y,r),RED);poly(c,triUp(x,y,r,.66),WHITE);txt(c,'!',x,y+r*.1,r*.7,BLK);break;}
    case 'works':{poly(c,triUp(x,y,r),RED);poly(c,triUp(x,y,r,.66),WHITE);disc(c,x-r*.06,y-r*.14,r*.07,BLK);poly(c,[[x-r*.12,y-r*.05],[x+r*.01,y-r*.05],[x+r*.04,y+r*.22],[x-r*.16,y+r*.22]],BLK);poly(c,[[x+r*.05,y+r*.3],[x+r*.25,y+r*.05],[x+r*.45,y+r*.3]],BLK);break;}
  }
}

/* ---------- recogniser: 32x32 crop -> normalised features -> template similarity -> softmax ---------- */
const maskIdx=[];
for(let y=0;y<32;y++)for(let x=0;x<32;x++){if(Math.hypot(x-15.5,y-15.5)<=MASKR)maskIdx.push(y*32+x);}
function featurize(img){
  const d=img.data,n=maskIdx.length*3,v=new Float32Array(n);
  let m=0,j=0;
  for(let i=0;i<maskIdx.length;i++){const o=maskIdx[i]*4;v[j++]=d[o];v[j++]=d[o+1];v[j++]=d[o+2];}
  for(j=0;j<n;j++)m+=v[j];
  m/=n;
  let s=0;
  for(j=0;j<n;j++){v[j]-=m;s+=v[j]*v[j];}
  s=Math.sqrt(s)||1;
  for(j=0;j<n;j++)v[j]/=s;
  return v;
}
function renderTemplate(k,q,bg){
  const cv=document.createElement('canvas');cv.width=cv.height=q;
  const c=cv.getContext('2d');
  c.fillStyle=bg;c.fillRect(0,0,q,q);
  const r=q/2/1.05;
  drawPole(c,q/2,q/2,r,q);drawSign(c,k,q/2,q/2,r);
  const out=document.createElement('canvas');out.width=out.height=32;
  const o=out.getContext('2d',{willReadFrequently:true});
  o.imageSmoothingQuality='high';o.drawImage(cv,0,0,32,32);
  return o.getImageData(0,0,32,32);
}
const TEMPL=CLASSES.map(function(k){
  const list=[];
  [32,22,14].forEach(function(q){['#6f8f63','#b9cfe2'].forEach(function(bg){list.push(featurize(renderTemplate(k,q,bg)));});});
  return list;
});
const crop=document.createElement('canvas');crop.width=crop.height=32;
const cctx=crop.getContext('2d',{willReadFrequently:true});
cctx.imageSmoothingQuality='high';
function classify(){
  const f=featurize(cctx.getImageData(0,0,32,32));
  const lg=TEMPL.map(function(list){
    let b=-1;
    for(let a=0;a<list.length;a++){const t=list[a];let s=0;for(let i=0;i<t.length;i++)s+=t[i]*f[i];if(s>b)b=s;}
    return b*SCALE;
  });
  const mx=Math.max.apply(null,lg);
  let sum=0;const e=lg.map(function(v){const z=Math.exp(v-mx);sum+=z;return z;});
  return e.map(function(z){return z/sum;});
}

/* ---------- simulator ---------- */
const $=function(id){return document.getElementById(id);};
const cam=$('cam'),dctx=cam.getContext('2d',{willReadFrequently:true});
const small=document.createElement('canvas'),sctx=small.getContext('2d');
const reduce=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const state={playing:!reduce,cond:'day',v:55,noise:5,blur:0,rate:.5,odo:0,gap:0,sign:null,lastCls:-1,limit:null,stats:{total:0,ok:0,dists:[]},log:[]};

const PAL={
  day:{skyT:'#5f9fdc',skyB:'#cfe4f4',hill:'#4a7a4f',grassFar:'#6a9a4c',grass:'#4a7636',road:'#62666b',line:'#f1f1ec',tree:'#2f5a2a',trunk:'#4b3a2a'},
  fog:{skyT:'#b9c0c6',skyB:'#dde1e4',hill:'#97a595',grassFar:'#a3ae9b',grass:'#8c9a85',road:'#8f9296',line:'#e6e6e2',tree:'#6d8268',trunk:'#7b7468'},
  rain:{skyT:'#6f7f8c',skyB:'#a7b2ba',hill:'#4a6a52',grassFar:'#58794c',grass:'#3f6136',road:'#4d5156',line:'#d7d9d6',tree:'#2a4a2a',trunk:'#3d3226'}
};
const TREEOFF=[9.6,11.2,10.4,12.0,9.9,11.6,10.8,12.4];
function ys(z){return HOR+4.2*K/z;}
function geom(d){const r=K/d;return {r:r,x:VPX+5.8*r,yG:HOR+4.2*r,y:HOR+0.9*r};}
function clamp(v,a,b){return Math.max(a,Math.min(b,v));}

function paintSign(alpha){
  const sg=state.sign;if(!sg)return;
  const g=geom(sg.d);
  dctx.save();dctx.globalAlpha=alpha;
  drawPole(dctx,g.x,g.y,g.r,g.yG-g.y);
  drawSign(dctx,CLASSES[sg.cls],g.x,g.y,g.r);
  dctx.restore();
}
function drawScene(){
  const S=state,c=dctx,p=PAL[S.cond==='night'?'day':S.cond];
  let g=c.createLinearGradient(0,0,0,HOR);g.addColorStop(0,p.skyT);g.addColorStop(1,p.skyB);
  c.fillStyle=g;c.fillRect(0,0,W,HOR+1);
  c.beginPath();c.moveTo(0,HOR+1);
  for(let x=0;x<=W;x+=16)c.lineTo(x,HOR-7-5*Math.sin(x*.031)-3*Math.sin(x*.117+1));
  c.lineTo(W,HOR+1);c.closePath();c.fillStyle=p.hill;c.fill();
  g=c.createLinearGradient(0,HOR,0,H);g.addColorStop(0,p.grassFar);g.addColorStop(1,p.grass);
  c.fillStyle=g;c.fillRect(0,HOR,W,H-HOR);
  const y0=HOR+2,s0=(y0-HOR)/4.2,s1=(H-HOR)/4.2;
  poly(c,[[VPX-5*s0,y0],[VPX+5*s0,y0],[VPX+5*s1,H],[VPX-5*s1,H]],p.road);
  [-1,1].forEach(function(sg){poly(c,[[VPX+sg*4.55*s0,y0],[VPX+sg*4.8*s0,y0],[VPX+sg*4.8*s1,H],[VPX+sg*4.55*s1,H]],p.line);});
  const per=9,off=S.odo%per;
  for(let k=0;k<14;k++){
    const za=4+k*per-off,zb=za+3;
    if(zb<5.7)continue;
    const a=Math.max(za,5.7),ya=ys(a),yb=ys(zb);
    if(ya>H+1)continue;
    const sa=(ya-HOR)/4.2,sb=(yb-HOR)/4.2;
    poly(c,[[VPX-.11*sb,yb],[VPX+.11*sb,yb],[VPX+.11*sa,ya],[VPX-.11*sa,ya]],p.line);
  }
  const trees=[];
  for(let k=0;k<8;k++){const z=(((k*11-S.odo)%88)+88)%88+6;trees.push({z:z,off:TREEOFF[k]});}
  trees.sort(function(a,b){return b.z-a.z;});
  trees.forEach(function(t){
    const s=K/t.z,x=VPX-t.off*s,yG=HOR+4.2*s;
    c.fillStyle=p.trunk;c.fillRect(x-.1*s,yG-2.2*s,.2*s,2.2*s);
    disc(c,x,yG-3.0*s,1.3*s,p.tree);
  });
  const d=S.sign?S.sign.d:999;
  c.beginPath();c.moveTo(0,H);c.lineTo(0,H-18);c.quadraticCurveTo(W/2,H-36,W,H-18);c.lineTo(W,H);c.closePath();
  c.fillStyle='rgba(20,22,25,.94)';c.fill();
  if(S.cond==='day'){paintSign(1);}
  else if(S.cond==='rain'){paintSign(.95);c.fillStyle='rgba(35,45,55,.22)';c.fillRect(0,0,W,H);
    c.strokeStyle='rgba(220,230,240,.32)';c.lineWidth=1;c.beginPath();
    for(let i=0;i<110;i++){const x=Math.random()*W,y=Math.random()*H;c.moveTo(x,y);c.lineTo(x-4,y+14);}
    c.stroke();}
  else if(S.cond==='fog'){
    g=c.createLinearGradient(0,0,0,H);
    g.addColorStop(0,'rgba(214,219,223,.7)');g.addColorStop(HOR/H,'rgba(214,219,223,.86)');g.addColorStop(1,'rgba(214,219,223,.35)');
    c.fillStyle=g;c.fillRect(0,0,W,H);
    paintSign(clamp(Math.exp(-d/38),.1,1));
  }
  else if(S.cond==='night'){
    paintSign(1);
    g=c.createRadialGradient(VPX+20,HOR+70,20,VPX+20,HOR+70,330);
    g.addColorStop(0,'rgba(4,8,22,.35)');g.addColorStop(.6,'rgba(4,8,22,.8)');g.addColorStop(1,'rgba(4,8,22,.93)');
    c.fillStyle=g;c.fillRect(0,0,W,H);
    const a=clamp((60-d)/50,0,.85);
    if(a>0.02)paintSign(a);
  }
}
function blurFrame(b){
  if(b<=.01)return;
  const f=1+b*.9,w=Math.max(8,Math.round(W/f)),h=Math.max(8,Math.round(H/f));
  small.width=w;small.height=h;
  sctx.drawImage(cam,0,0,w,h);
  dctx.drawImage(small,0,0,w,h,0,0,W,H);
}
function addNoise(sig){
  if(sig<=0)return;
  const im=dctx.getImageData(0,0,W,H),d=im.data,amp=sig*2.2;
  for(let i=0;i<d.length;i+=4){const n=(Math.random()+Math.random()-1)*amp;d[i]+=n;d[i+1]+=n;d[i+2]+=n;}
  dctx.putImageData(im,0,0);
}

/* ---------- ADAS logic ---------- */
function advisory(k,v){
  switch(k.kind){
    case 'speed':{const over=v-k.val,t='Speed limit '+k.val;
      if(over>15)return ['crit',t,'You are driving '+v+' km/h, '+over+' km/h over the limit. Slow down now.'];
      if(over>0)return ['warn',t,'You are '+over+' km/h over the limit. Ease off the accelerator.'];
      return ['ok',t,'Driving '+v+' km/h. You are within the limit.'];}
    case 'stop':return v>5?['crit','Stop sign ahead','Brake and come to a full stop at the line.']:['ok','Stop sign ahead','Vehicle is stopped. Check cross traffic before moving.'];
    case 'yield':return ['warn','Give way ahead','Slow down and give way to cross traffic.'];
    case 'noentry':return ['crit','No entry','Do not enter this road. Find another route.'];
    case 'priority':return ['ok','Priority road','You have right of way at the junctions ahead.'];
    case 'ahead':return ['info','Ahead only','Only straight movement is allowed here.'];
    case 'right':return ['info','Turn right ahead','Prepare to turn right at the junction.'];
    case 'caution':return ['warn','General caution','Hazard ahead. Reduce speed and stay alert.'];
    case 'works':return ['warn','Road work','Slow down. Workers or machinery may be on the road.'];
  }
  return ['idle','',''];
}

/* ---------- sign lifecycle ---------- */
function spawn(d){
  const sel=$('sel-sign').value;let c;
  if(sel==='random'){do{c=Math.floor(Math.random()*CLASSES.length);}while(c===state.lastCls);}
  else c=+sel;
  state.lastCls=c;
  state.sign={cls:c,d:d,tr:{p:null,top:-1,count:0,pt:0,confirmed:false,last:null}};
}
function stamp(){
  const n=new Date(),z=function(v){return String(v).padStart(2,'0');};
  return z(n.getHours())+':'+z(n.getMinutes())+':'+z(n.getSeconds());
}
function endSign(){
  const S=state,sg=S.sign,tr=sg.tr,last=tr.last;
  S.stats.total++;
  const ok=!!last&&last.cls===sg.cls;
  if(ok)S.stats.ok++;
  if(last)S.stats.dists.push(last.d);
  S.log.unshift({t:stamp(),truth:CLASSES[sg.cls].name,pred:last?CLASSES[last.cls].name:'Not confirmed',conf:last?last.p:null,dist:last?last.d:null,ok:ok});
  if(S.log.length>8)S.log.pop();
  S.sign=null;
  renderLog();
}
function renderLog(){
  const S=state,tb=$('log');
  tb.textContent='';
  if(!S.log.length){const tr=document.createElement('tr'),td=document.createElement('td');td.className='empty';td.colSpan=6;td.textContent='Waiting for the first sign to pass. Each pass adds a row here.';tr.appendChild(td);tb.appendChild(tr);}
  S.log.forEach(function(e){
    const tr=document.createElement('tr');
    const cells=[[e.t,'mono'],[e.truth,''],[e.pred,''],[e.conf===null?'n/a':(e.conf*100).toFixed(0)+'%','num mono'],[e.dist===null?'n/a':e.dist.toFixed(1)+' m','num mono'],[e.ok?'Correct':'Wrong',e.ok?'res-ok':'res-bad']];
    cells.forEach(function(c){const td=document.createElement('td');td.textContent=c[0];if(c[1])td.className=c[1];tr.appendChild(td);});
    tb.appendChild(tr);
  });
  const st=S.stats;
  $('st-total').textContent=st.total;
  $('st-ok').textContent=st.ok;
  $('st-acc').textContent=st.total?Math.round(st.ok/st.total*100)+'%':'n/a';
  $('st-dist').textContent=st.dists.length?(st.dists.reduce(function(a,b){return a+b;},0)/st.dists.length).toFixed(1)+' m':'n/a';
}

/* ---------- per-frame tracking ---------- */
function track(probs){
  const sg=state.sign,tr=sg.tr;
  if(!tr.p)tr.p=probs.slice();
  else for(let i=0;i<probs.length;i++)tr.p[i]=tr.p[i]*.55+probs[i]*.45;
  let top=0;for(let i=1;i<tr.p.length;i++)if(tr.p[i]>tr.p[top])top=i;
  if(top===tr.top)tr.count++;else{tr.top=top;tr.count=1;}
  tr.pt=tr.p[top];
  tr.confirmed=tr.count>=NCONF&&tr.pt>=PTHR;
  if(tr.confirmed){
    if(!tr.last||tr.last.cls!==top)tr.last={cls:top,p:tr.pt,d:sg.d};else tr.last.p=tr.pt;
    if(CLASSES[top].kind==='speed')state.limit=CLASSES[top].val;
  }
}

/* ---------- HUD on the camera frame ---------- */
function chipText(c,t,x,y,align,fill,font){
  c.font=font;c.textBaseline='middle';
  const w=c.measureText(t).width+14;
  const bx=align==='right'?x-w:x;
  c.fillStyle='rgba(0,0,0,.5)';c.fillRect(bx,y-11,w,22);
  c.fillStyle=fill;c.textAlign='left';c.fillText(t,bx+7,y);
}
function drawHUD(now,geo,mode){
  const c=dctx,S=state,mono='500 12px "IBM Plex Mono",monospace';
  c.save();
  const d=new Date(),z=function(v){return String(v).padStart(2,'0');};
  const ds=d.getFullYear()+'-'+z(d.getMonth()+1)+'-'+z(d.getDate())+' '+stamp();
  c.fillStyle='rgba(0,0,0,.5)';c.fillRect(8,8,172,22);
  if(Math.floor(now/600)%2===0){disc(c,20,19,4,'#ff4040');}
  c.font=mono;c.textAlign='left';c.textBaseline='middle';c.fillStyle='#fff';c.fillText(ds,30,19);
  chipText(c,S.v+' km/h',W-8,19,'right','#fff',mono);
  chipText(c,S.sign?'Sign '+S.sign.d.toFixed(1)+' m':'No sign in view',8,H-44,'left','#fff',mono);
  chipText(c,S.cond.toUpperCase(),W-8,H-44,'right','#fff',mono);
  if(geo&&mode!=='scan'){
    const tr=S.sign.tr,col=mode==='conf'?'#34c27a':'#f0b43a';
    const half=geo.r*1.25,x0=geo.x-half,y0=geo.y-half,x1=geo.x+half,y1=geo.y+half,L=Math.max(5,half*.4);
    c.strokeStyle=col;c.lineWidth=2;c.beginPath();
    c.moveTo(x0,y0+L);c.lineTo(x0,y0);c.lineTo(x0+L,y0);
    c.moveTo(x1-L,y0);c.lineTo(x1,y0);c.lineTo(x1,y0+L);
    c.moveTo(x1,y1-L);c.lineTo(x1,y1);c.lineTo(x1-L,y1);
    c.moveTo(x0+L,y1);c.lineTo(x0,y1);c.lineTo(x0,y1-L);
    c.stroke();
    const label=CLASSES[tr.top].name+'  '+Math.round(tr.pt*100)+'%';
    c.font='600 14px "Barlow Condensed","Arial Narrow",Arial,sans-serif';
    const w=c.measureText(label).width+14,lx=clamp(x0,4,W-w-4),ly=Math.max(4,y0-24);
    c.fillStyle=col;c.fillRect(lx,ly,w,20);
    c.fillStyle='#10161c';c.textAlign='left';c.textBaseline='middle';c.fillText(label,lx+7,ly+10);
  }
  c.restore();
}

/* ---------- panel ---------- */
const miniC=$('mini'),mctx=miniC.getContext('2d');
const s1=$('s1').getContext('2d'),s2=$('s2').getContext('2d'),s3=$('s3').getContext('2d');
const probsEl=$('probs'),pipsEl=$('pips'),rows=[],pips=[];
for(let i=0;i<5;i++){
  const li=document.createElement('li');
  li.innerHTML='<span class="p-name"></span><span class="p-bar"><i></i></span><span class="p-pct"></span>';
  probsEl.appendChild(li);
  rows.push({name:li.children[0],bar:li.children[1].firstChild,pct:li.children[2]});
}
for(let i=0;i<NCONF;i++){const s=document.createElement('span');pipsEl.appendChild(s);pips.push(s);}
const cache={};
function setText(id,v){if(cache[id]!==v){cache[id]=v;$(id).textContent=v;}}
function drawMini(k,alpha){
  mctx.clearRect(0,0,96,96);
  if(!k){mctx.strokeStyle='#8a99a8';mctx.lineWidth=2;mctx.setLineDash([5,5]);mctx.beginPath();mctx.arc(48,48,38,0,Math.PI*2);mctx.stroke();mctx.setLineDash([]);return;}
  mctx.globalAlpha=alpha;drawSign(mctx,k,48,48,40);mctx.globalAlpha=1;
}
let miniKey='';
function updateUI(mode,geo){
  const S=state,sg=S.sign,tr=sg?sg.tr:null;
  const chip=$('chip');
  const chipS=mode==='conf'?'conf':mode==='cand'?'cand':'idle';
  if(chip.dataset.s!==chipS)chip.dataset.s=chipS;
  setText('chip',mode==='conf'?'Confirmed':mode==='cand'?'Verifying':'Scanning');
  let sev='idle',title,body,mk=null,ma=1;
  if(mode==='conf'){const k=CLASSES[tr.top],a=advisory(k,S.v);sev=a[0];title=a[1];body=a[2];mk=k;}
  else if(mode==='cand'){title='Verifying sign';body='Best match is '+CLASSES[tr.top].name+' at '+Math.round(tr.pt*100)+'%. No alert until the match is stable.';mk=CLASSES[tr.top];ma=.35;}
  else{title='Scanning the road';body=sg?'Sign in view, '+sg.d.toFixed(1)+' m away. Too far to classify yet.':'No sign ahead.';}
  const al=$('alert');if(al.dataset.sev!==sev)al.dataset.sev=sev;
  setText('a-title',title);setText('a-body',body);
  const key=(mk?CLASSES.indexOf(mk):-1)+'|'+ma;
  if(key!==miniKey){miniKey=key;drawMini(mk,ma);}
  setText('v-now',S.v+' km/h');
  const fill=$('v-fill');fill.style.width=Math.min(100,S.v/130*100)+'%';
  let st='';
  if(S.limit!==null){const over=S.v-S.limit;st=over>15?'crit':over>0?'warn':'ok';
    setText('v-lim','Active limit '+S.limit+' km/h');
    const mark=$('v-mark');mark.hidden=false;mark.style.left=(S.limit/130*100)+'%';}
  else{setText('v-lim','No limit seen yet');$('v-mark').hidden=true;}
  if(fill.dataset.s!==st)fill.dataset.s=st;
  setText('dist-read','Distance '+(sg?sg.d.toFixed(1):'none')+(sg?' m':''));
  const sorted=[];
  if(tr&&tr.p){for(let i=0;i<tr.p.length;i++)sorted.push([i,tr.p[i]]);sorted.sort(function(a,b){return b[1]-a[1];});}
  for(let i=0;i<5;i++){
    const r=rows[i],e=sorted[i];
    if(e){r.name.textContent=CLASSES[e[0]].name;r.bar.style.width=(e[1]*100).toFixed(1)+'%';r.pct.textContent=(e[1]*100).toFixed(0)+'%';}
    else{r.name.textContent='n/a';r.bar.style.width='0';r.pct.textContent='0%';}
  }
  const n=mode==='scan'||!tr?0:Math.min(NCONF,tr.count);
  for(let i=0;i<NCONF;i++){const on=i<n;if(pips[i].classList.contains('on')!==on)pips[i].classList.toggle('on',on);}
  setText('s4',mode==='conf'||mode==='cand'?CLASSES[tr.top].name:'No sign in range');
  setText('s4p',mode==='conf'||mode==='cand'?'Top score '+Math.round(tr.pt*100)+'% across '+CLASSES.length+' classes.':'Softmax scores over the sign classes.');
  const dr=$('in-dist');
  if(sg&&document.activeElement!==dr){dr.value=sg.d;setText('out-dist',sg.d.toFixed(1)+' m');}
}

/* ---------- main tick ---------- */
function step(dt){
  const S=state;
  if(!S.playing)return;
  const mps=S.v/3.6*S.rate;
  S.odo+=mps*dt;
  if(S.sign){S.sign.d-=mps*dt;if(S.sign.d<=PASS_D){endSign();S.gap=.9;}}
  else{S.gap-=dt;if(S.gap<=0)spawn(60);}
}
function tick(dt,now){
  step(dt);
  drawScene();
  blurFrame(state.blur+(state.cond==='rain'?.6:0));
  addNoise(state.noise);
  let mode='scan',geo=null;
  if(state.sign){
    geo=geom(state.sign.d);
    if(geo.r>=DET_R){
      const side=2.1*geo.r;
      cctx.clearRect(0,0,32,32);
      cctx.drawImage(cam,geo.x-side/2,geo.y-side/2,side,side,0,0,32,32);
      track(classify());
      s2.drawImage(cam,geo.x-side*.65,geo.y-side*.65,side*1.3,side*1.3,0,0,64,64);
      s3.globalCompositeOperation='source-over';
      s3.clearRect(0,0,32,32);s3.drawImage(crop,0,0);
      s3.globalCompositeOperation='destination-in';
      s3.beginPath();s3.arc(16,16,MASKR+.5,0,Math.PI*2);s3.fill();
      s3.globalCompositeOperation='source-over';
      mode=state.sign.tr.confirmed?'conf':'cand';
    }else{
      state.sign.tr={p:null,top:-1,count:0,pt:0,confirmed:false,last:state.sign.tr.last};
      s2.clearRect(0,0,64,64);s3.clearRect(0,0,32,32);
    }
  }else{s2.clearRect(0,0,64,64);s3.clearRect(0,0,32,32);}
  drawHUD(now,geo,mode);
  s1.drawImage(cam,0,0,W,H,0,0,160,90);
  updateUI(mode,geo);
}
let acc=0,prev=performance.now();
function loop(now){
  const dt=(now-prev)/1000;prev=now;acc+=dt;
  if(acc>=1/30-.002){const d=Math.min(acc,.1);acc=0;tick(d,now);}
  requestAnimationFrame(loop);
}

/* ---------- controls ---------- */
const sel=$('sel-sign');
CLASSES.forEach(function(k,i){const o=document.createElement('option');o.value=i;o.textContent=k.name;sel.appendChild(o);});
sel.addEventListener('change',function(){spawn(34);});
$('cond').addEventListener('click',function(e){
  const b=e.target.closest('button');if(!b)return;
  state.cond=b.dataset.c;
  Array.prototype.forEach.call($('cond').children,function(x){x.setAttribute('aria-pressed',x===b?'true':'false');});
});
$('in-speed').addEventListener('input',function(e){state.v=+e.target.value;$('out-speed').textContent=state.v+' km/h';});
$('in-noise').addEventListener('input',function(e){state.noise=+e.target.value;$('out-noise').textContent=e.target.value;});
$('in-blur').addEventListener('input',function(e){state.blur=+e.target.value;$('out-blur').textContent=e.target.value;});
$('in-dist').addEventListener('input',function(e){const v=+e.target.value;if(!state.sign)spawn(v);state.sign.d=v;$('out-dist').textContent=v.toFixed(1)+' m';});
$('sel-rate').addEventListener('change',function(e){state.rate=+e.target.value;});
$('btn-play').addEventListener('click',function(){state.playing=!state.playing;this.textContent=state.playing?'Pause':'Play';});
$('btn-next').addEventListener('click',function(){spawn(60);});
if(!state.playing)$('btn-play').textContent='Play';

renderLog();
spawn(state.playing?30:18);
requestAnimationFrame(loop);
})();
