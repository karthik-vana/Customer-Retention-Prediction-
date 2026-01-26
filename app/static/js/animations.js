/**
 * ANIMATION MANAGER
 * Handles page transitions, loading states, and smooth interactions
 */

class AnimationManager {
    constructor() {
        this.loadingBar = document.getElementById('loading-bar');
        this.currentProgress = 0;
    }

    /**
     * Animate loading bar
     */
    startLoading() {
        this.currentProgress = 10;
        this.loadingBar.style.width = this.currentProgress + '%';
        
        const interval = setInterval(() => {
            if (this.currentProgress < 90) {
                this.currentProgress += Math.random() * 20;
                this.loadingBar.style.width = Math.min(this.currentProgress, 90) + '%';
            } else {
                clearInterval(interval);
            }
        }, 300);
        
        return interval;
    }

    completeLoading() {
        this.currentProgress = 100;
        this.loadingBar.style.width = '100%';
        
        setTimeout(() => {
            this.loadingBar.style.opacity = '0';
            this.loadingBar.style.transition = 'opacity 0.5s ease';
        }, 500);
        
        setTimeout(() => {
            this.loadingBar.style.width = '0%';
            this.loadingBar.style.opacity = '1';
            this.loadingBar.style.transition = '';
        }, 1000);
    }

    /**
     * Animate elements with stagger effect
     */
    staggerElements(selector, baseDelay = 0.1) {
        const elements = document.querySelectorAll(selector);
        elements.forEach((el, index) => {
            el.style.animation = `fadeInUp 0.6s ease-out forwards`;
            el.style.animationDelay = (baseDelay * (index + 1)) + 's';
        });
    }

    /**
     * Animate page transition
     */
    transitionPage(callback) {
        const main = document.querySelector('main');
        main.style.animation = 'fadeOut 0.3s ease-out';
        
        setTimeout(() => {
            callback();
            main.style.animation = '';
        }, 300);
    }

    /**
     * Smooth scroll to element
     */
    smoothScroll(selector, offset = 100) {
        const element = document.querySelector(selector);
        if (element) {
            const top = element.getBoundingClientRect().top + window.scrollY - offset;
            window.scrollTo({
                top: top,
                behavior: 'smooth'
            });
        }
    }

    /**
     * Create skeleton loading
     */
    createSkeleton(count = 1) {
        const skeletons = [];
        for (let i = 0; i < count; i++) {
            const skeleton = document.createElement('div');
            skeleton.className = 'shimmer rounded-lg';
            skeleton.style.height = '100px';
            skeleton.style.marginBottom = '15px';
            skeletons.push(skeleton);
        }
        return skeletons;
    }

    /**
     * Animate number counter
     */
    countUp(element, target, duration = 2000) {
        const start = 0;
        const increment = target / (duration / 16);
        let current = start;

        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            element.textContent = Math.round(current);
        }, 16);
    }

    /**
     * Ripple effect on click
     */
    addRippleEffect(element) {
        element.addEventListener('click', (e) => {
            const ripple = document.createElement('span');
            const rect = element.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;

            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.className = 'ripple';
            ripple.style.cssText += `
                position: absolute;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.6);
                transform: scale(0);
                animation: ripple 0.6s ease-out;
                pointer-events: none;
            `;

            element.style.position = 'relative';
            element.style.overflow = 'hidden';
            element.appendChild(ripple);

            setTimeout(() => ripple.remove(), 600);
        });
    }
}

// Global instance
window.animator = new AnimationManager();

// Ripple animation styles
const rippleStyle = document.createElement('style');
rippleStyle.textContent = `
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    @keyframes fadeOut {
        to {
            opacity: 0;
        }
    }
`;
document.head.appendChild(rippleStyle);

// Initialize animations on page load
document.addEventListener('DOMContentLoaded', () => {
    // Add stagger animation to cards
    animator.staggerElements('[data-aos]', 0.08);
    
    // Add ripple effect to buttons
    document.querySelectorAll('button').forEach(btn => {
        animator.addRippleEffect(btn);
    });
});

// Smooth scroll behavior
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        animator.smoothScroll(this.getAttribute('href'));
    });
});
