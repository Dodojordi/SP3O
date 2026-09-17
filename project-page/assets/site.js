const nav=document.querySelector('[data-nav]');
const toggle=document.querySelector('[data-nav-toggle]');
toggle?.addEventListener('click',()=>{const open=nav.classList.toggle('open');toggle.setAttribute('aria-expanded',String(open))});
nav?.querySelectorAll('a').forEach(link=>link.addEventListener('click',()=>{nav.classList.remove('open');toggle?.setAttribute('aria-expanded','false')}));

const copy=document.querySelector('[data-copy-bib]');
copy?.addEventListener('click',async()=>{const text=document.querySelector('#bibtex')?.textContent??'';try{await navigator.clipboard.writeText(text)}catch{const range=document.createRange(),selection=getSelection();range.selectNodeContents(document.querySelector('#bibtex'));selection.removeAllRanges();selection.addRange(range);document.execCommand('copy');selection.removeAllRanges()}const card=copy.closest('.code-card');card.classList.add('copied');setTimeout(()=>card.classList.remove('copied'),1800)});

const dialog=document.querySelector('[data-lightbox-dialog]');
const dialogImage=dialog?.querySelector('img');
const dialogCaption=dialog?.querySelector('p');
document.querySelectorAll('[data-lightbox]').forEach(figure=>figure.addEventListener('click',()=>{const image=figure.querySelector('img');if(!image||!dialog)return;dialogImage.src=image.src;dialogImage.alt=image.alt;dialogCaption.textContent=figure.querySelector('figcaption')?.textContent??image.alt;dialog.showModal()}));
dialog?.querySelector('button')?.addEventListener('click',()=>dialog.close());
dialog?.addEventListener('click',event=>{if(event.target===dialog)dialog.close()});
