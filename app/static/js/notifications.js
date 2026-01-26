/**
 * TOAST NOTIFICATIONS SYSTEM
 * Modern, smooth notifications for user feedback
 * Supports: Success, Error, Warning, Info
 */

class ToastNotification {
    constructor() {
        this.container = document.getElementById('toast-container');
    }

    /**
     * Show a toast notification
     * @param {string} message - Notification message
     * @param {string} type - success|error|warning|info
     * @param {number} duration - Display duration in ms (0 = persistent)
     */
    show(message, type = 'info', duration = 4000) {
        const toast = this.createToast(message, type);
        this.container.appendChild(toast);

        // Animate in
        setTimeout(() => {
            toast.classList.add('fade-in-up');
        }, 10);

        // Auto remove
        if (duration > 0) {
            setTimeout(() => {
                this.remove(toast);
            }, duration);
        }

        return toast;
    }

    createToast(message, type) {
        const toast = document.createElement('div');
        const icons = {
            success: '✓',
            error: '✕',
            warning: '⚠',
            info: 'ℹ'
        };
        
        const colors = {
            success: 'bg-green-100 border-green-400 text-green-800',
            error: 'bg-red-100 border-red-400 text-red-800',
            warning: 'bg-yellow-100 border-yellow-400 text-yellow-800',
            info: 'bg-blue-100 border-blue-400 text-blue-800'
        };

        toast.className = `p-4 mb-3 rounded-lg border-l-4 shadow-lg flex items-center gap-3 ${colors[type]} toast-notification`;
        toast.style.cssText = `
            animation-duration: 0.4s;
            min-width: 300px;
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        `;

        toast.innerHTML = `
            <span class="text-xl font-bold">${icons[type]}</span>
            <span class="flex-1">${message}</span>
            <button onclick="this.parentElement.remove()" class="ml-2 font-bold hover:opacity-70 transition">×</button>
        `;

        return toast;
    }

    remove(toast) {
        toast.style.animation = 'fadeOut 0.4s ease-out forwards';
        setTimeout(() => toast.remove(), 400);
    }

    success(message, duration) {
        return this.show(message, 'success', duration);
    }

    error(message, duration) {
        return this.show(message, 'error', duration);
    }

    warning(message, duration) {
        return this.show(message, 'warning', duration);
    }

    info(message, duration) {
        return this.show(message, 'info', duration);
    }

    loading(message) {
        const toast = document.createElement('div');
        toast.className = 'p-4 mb-3 rounded-lg border-l-4 border-blue-400 bg-blue-100 text-blue-800 shadow-lg flex items-center gap-3 toast-notification';
        toast.innerHTML = `
            <div class="spin w-5 h-5 border-2 border-blue-400 border-t-transparent rounded-full"></div>
            <span>${message}</span>
        `;
        this.container.appendChild(toast);
        
        setTimeout(() => {
            toast.classList.add('fade-in-up');
        }, 10);
        
        return toast;
    }
}

// Global instance
window.toast = new ToastNotification();

// Fade out animation
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(400px);
        }
    }
`;
document.head.appendChild(style);
