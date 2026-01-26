// Footer drawer toggle
(function(){
    const drawer = document.getElementById('footerDrawer');
    const toggle = document.getElementById('footerToggle');
    const inner = document.getElementById('footerInner');

    if(!drawer || !toggle || !inner) return;

    // restore state
    const openState = localStorage.getItem('footer_open') === '1';
    if(openState) drawer.classList.add('open');
    inner.setAttribute('aria-hidden', openState ? 'false' : 'true');
    toggle.setAttribute('aria-expanded', openState ? 'true' : 'false');

    toggle.addEventListener('click', ()=>{
        const isOpen = drawer.classList.toggle('open');
        inner.setAttribute('aria-hidden', isOpen ? 'false' : 'true');
        toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
        localStorage.setItem('footer_open', isOpen ? '1' : '0');
        // animate focus
        if(isOpen) inner.querySelector('.footer-name')?.focus();
    });

    // close on escape
    document.addEventListener('keydown', (e)=>{
        if(e.key === 'Escape' && drawer.classList.contains('open')){
            drawer.classList.remove('open');
            inner.setAttribute('aria-hidden','true');
            toggle.setAttribute('aria-expanded','false');
            localStorage.setItem('footer_open','0');
        }
    });
})();
