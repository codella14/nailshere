/**
 * Hand Tracking and Nail Overlay System
 * Provides real-time hand detection and nail positioning for AR try-on
 */

class HandTracker {
    constructor() {
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.segmentedImage = null;
        this.isActive = false;
        this.animationId = null;
        this.handPosition = { x: 0, y: 0, scale: 1, rotation: 0 };
        this.fingerData = {
            tip: { x: 0, y: 0 },
            base: { x: 0, y: 0 },
            angle: 0,
            confidence: 0
        };
        this.settings = {
            opacity: 0.8,
            scale: 1.0,
            rotation: 0,
            smoothing: 0.1,
            position: 'center',
            showHandOutline: true,
            showFingerTracking: true,
            fingerDetectionEnabled: true
        };
        this.lastFingerDetection = 0;
        this.fingerDetectionInterval = 150; // ms between detections (increased for stability)
        this.positionHistory = [];
        this.maxHistoryLength = 5;
        this.unstableCount = 0;
        this.maxUnstableCount = 10;
    }

    init(videoElement, canvasElement, segmentedImageUrl) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
        
        // Load segmented image
        this.segmentedImage = new Image();
        this.segmentedImage.crossOrigin = "anonymous";
        this.segmentedImage.onload = () => {
            console.log('Segmented image loaded for hand tracking');
        };
        this.segmentedImage.onerror = () => {
            console.error('Failed to load segmented image for hand tracking');
        };
        this.segmentedImage.src = segmentedImageUrl;
    }

    start() {
        this.isActive = true;
        this.track();
    }

    stop() {
        this.isActive = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }

    updateSettings(newSettings) {
        this.settings = { ...this.settings, ...newSettings };
    }

    track() {
        if (!this.isActive || !this.video || !this.canvas) return;

        // Set canvas size to match video
        this.canvas.width = this.video.videoWidth || this.video.clientWidth;
        this.canvas.height = this.video.videoHeight || this.video.clientHeight;

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Detect hand position (simplified approach)
        this.detectHandPosition();

        // Draw nail overlay
        this.drawNailOverlay();

        // Continue tracking
        this.animationId = requestAnimationFrame(() => this.track());
    }

    detectHandPosition() {
        if (this.settings.fingerDetectionEnabled) {
            // Limit finger detection frequency for performance
            const now = Date.now();
            if (now - this.lastFingerDetection > this.fingerDetectionInterval) {
                // Try finger detection
                const fingerDetected = this.detectFinger();
                if (fingerDetected && this.fingerData.confidence > 0.4) {
                    // Check if the new position is stable (not a sudden jump)
                    const newX = this.fingerData.tip.x;
                    const newY = this.fingerData.tip.y;
                    
                    if (this.isPositionStable(newX, newY)) {
                        // Add to position history
                        this.positionHistory.push({ x: newX, y: newY, time: now });
                        if (this.positionHistory.length > this.maxHistoryLength) {
                            this.positionHistory.shift();
                        }
                        
                        // Use average position for stability
                        const avgPos = this.getAveragePosition();
                        
                        // Smooth the finger position to prevent glitching
                        const smoothingFactor = 0.4;
                        this.handPosition.x += (avgPos.x - this.handPosition.x) * smoothingFactor;
                        this.handPosition.y += (avgPos.y - this.handPosition.y) * smoothingFactor;
                        
                        // Use finger angle for rotation only if confidence is high
                        if (this.settings.rotation === 0 && this.fingerData.confidence > 0.6) {
                            this.handPosition.rotation += (this.fingerData.angle - this.handPosition.rotation) * 0.15;
                        }
                        
                        // Scale based on finger length with smoothing
                        const fingerLength = Math.sqrt(
                            Math.pow(this.fingerData.tip.x - this.fingerData.base.x, 2) + 
                            Math.pow(this.fingerData.tip.y - this.fingerData.base.y, 2)
                        );
                        const targetScale = Math.min(Math.max(fingerLength / 120, 0.6), 1.8);
                        this.handPosition.scale += (targetScale - this.handPosition.scale) * 0.1;
                        
                        // Reset unstable count on successful detection
                        this.unstableCount = Math.max(0, this.unstableCount - 1);
                    } else {
                        // Position is unstable, increment counter
                        this.unstableCount++;
                        if (this.unstableCount >= this.maxUnstableCount) {
                            // Show warning and suggest disabling finger detection
                            this.showTrackingWarning();
                        }
                    }
                    
                    this.lastFingerDetection = now;
                    return;
                }
                this.lastFingerDetection = now;
            }
        }
        
        // Fallback to hand detection
        const detectedPosition = this.detectHandByColor();
        
        // If no hand detected, use center position
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        let targetX = detectedPosition.x;
        let targetY = detectedPosition.y;
        
        // Apply position offset based on settings
        const offset = 80; // pixels
        switch (this.settings.position) {
            case 'left':
                targetX -= offset;
                break;
            case 'right':
                targetX += offset;
                break;
            case 'top':
                targetY -= offset;
                break;
            case 'bottom':
                targetY += offset;
                break;
            case 'center':
            default:
                // No offset
                break;
        }

        // Smooth the position
        this.handPosition.x += (targetX - this.handPosition.x) * this.settings.smoothing;
        this.handPosition.y += (targetY - this.handPosition.y) * this.settings.smoothing;
        this.handPosition.scale = this.settings.scale;
    }

    drawNailOverlay() {
        if (!this.segmentedImage || !this.segmentedImage.complete) return;

        const ctx = this.ctx;
        const img = this.segmentedImage;
        
        // Ensure position is within canvas bounds
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        
        // Clamp position to canvas bounds
        const clampedX = Math.max(50, Math.min(canvasWidth - 50, this.handPosition.x));
        const clampedY = Math.max(50, Math.min(canvasHeight - 50, this.handPosition.y));
        
        // Calculate nail dimensions with bounds checking
        const nailWidth = Math.max(20, Math.min(200, img.width * this.handPosition.scale));
        const nailHeight = Math.max(20, Math.min(200, img.height * this.handPosition.scale));
        
        // Calculate position (center on hand position)
        const nailX = clampedX - nailWidth / 2;
        const nailY = clampedY - nailHeight / 2;

        // Save context
        ctx.save();

        // Draw hand outline if enabled
        if (this.settings.showHandOutline) {
            this.drawHandOutline();
        }

        // Draw finger tracking if enabled
        if (this.settings.showFingerTracking && this.fingerData.confidence > 0.3) {
            this.drawFingerTracking();
        }

        // Apply rotation if needed
        const rotationAngle = this.settings.rotation !== 0 ? this.settings.rotation : this.handPosition.rotation;
        if (Math.abs(rotationAngle) > 0.1) {
            // Move to center of nail for rotation
            ctx.translate(clampedX, clampedY);
            // Apply rotation (convert degrees to radians)
            ctx.rotate((rotationAngle * Math.PI) / 180);
            // Move back to nail position
            ctx.translate(-clampedX, -clampedY);
        }

        // Apply opacity
        ctx.globalAlpha = this.settings.opacity;

        // Draw the segmented nail
        ctx.drawImage(img, nailX, nailY, nailWidth, nailHeight);

        // Restore context
        ctx.restore();
    }

    drawHandOutline() {
        const ctx = this.ctx;
        const centerX = this.handPosition.x;
        const centerY = this.handPosition.y;
        
        // Draw a simple hand outline (oval shape)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        
        const handWidth = 120;
        const handHeight = 160;
        
        ctx.beginPath();
        ctx.ellipse(centerX, centerY, handWidth / 2, handHeight / 2, 0, 0, 2 * Math.PI);
        ctx.stroke();
        
        // Reset line dash
        ctx.setLineDash([]);
    }

    drawFingerTracking() {
        const ctx = this.ctx;
        const tip = this.fingerData.tip;
        const base = this.fingerData.base;
        const confidence = this.fingerData.confidence;
        
        // Draw finger line
        ctx.strokeStyle = `rgba(0, 255, 0, ${0.3 + confidence * 0.7})`;
        ctx.lineWidth = 3;
        ctx.setLineDash([]);
        
        ctx.beginPath();
        ctx.moveTo(tip.x, tip.y);
        ctx.lineTo(base.x, base.y);
        ctx.stroke();
        
        // Draw finger tip circle
        ctx.fillStyle = `rgba(0, 255, 0, ${0.5 + confidence * 0.5})`;
        ctx.beginPath();
        ctx.arc(tip.x, tip.y, 8, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw finger base circle
        ctx.fillStyle = `rgba(255, 0, 0, ${0.5 + confidence * 0.5})`;
        ctx.beginPath();
        ctx.arc(base.x, base.y, 6, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw confidence indicator
        ctx.fillStyle = `rgba(255, 255, 255, ${0.8})`;
        ctx.font = '12px Arial';
        ctx.fillText(`Finger: ${Math.round(confidence * 100)}%`, tip.x + 15, tip.y - 10);
    }

    // Finger detection using contour analysis
    detectFinger() {
        if (!this.video || !this.canvas) return false;

        // Use lower resolution for better performance
        const scale = 0.5;
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = (this.video.videoWidth || this.video.clientWidth) * scale;
        tempCanvas.height = (this.video.videoHeight || this.video.clientHeight) * scale;

        // Draw current video frame at reduced resolution
        tempCtx.drawImage(this.video, 0, 0, tempCanvas.width, tempCanvas.height);

        // Get image data
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const data = imageData.data;

        // Create binary mask for skin detection
        const width = tempCanvas.width;
        const height = tempCanvas.height;
        const skinMask = new Array(width * height).fill(0);

        // Sample every 2nd pixel for better performance
        for (let i = 0; i < data.length; i += 8) { // Skip every other pixel
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            
            if (this.isSkinColor(r, g, b)) {
                const pixelIndex = i / 4;
                skinMask[pixelIndex] = 1;
            }
        }

        // Find contours and detect finger-like shapes
        const finger = this.findFingerContour(skinMask, width, height);
        
        if (finger) {
            // Scale back up to original resolution
            this.fingerData = {
                tip: { x: finger.tip.x / scale, y: finger.tip.y / scale },
                base: { x: finger.base.x / scale, y: finger.base.y / scale },
                angle: finger.angle,
                confidence: finger.confidence
            };
            return true;
        }

        return false;
    }

    // Find finger contour using edge detection and shape analysis
    findFingerContour(skinMask, width, height) {
        // Find the largest connected component (hand)
        const visited = new Array(width * height).fill(false);
        let largestComponent = [];
        let maxSize = 0;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const index = y * width + x;
                if (skinMask[index] === 1 && !visited[index]) {
                    const component = this.floodFill(skinMask, visited, x, y, width, height);
                    if (component.length > maxSize) {
                        maxSize = component.length;
                        largestComponent = component;
                    }
                }
            }
        }

        if (largestComponent.length < 200) return null; // Increased minimum size

        // Find the finger tip (highest point in the component)
        const fingerTip = largestComponent.reduce((tip, point) => 
            point.y < tip.y ? point : tip
        );

        // Find finger base (lowest point in the component)
        const fingerBase = largestComponent.reduce((base, point) => 
            point.y > base.y ? point : base
        );

        // Calculate finger length and validate
        const fingerLength = Math.sqrt(
            Math.pow(fingerTip.x - fingerBase.x, 2) + 
            Math.pow(fingerTip.y - fingerBase.y, 2)
        );

        // Reject if finger is too short or too long
        if (fingerLength < 30 || fingerLength > 200) return null;

        // Calculate finger angle
        const angle = Math.atan2(fingerTip.x - fingerBase.x, fingerTip.y - fingerBase.y) * 180 / Math.PI;

        // Calculate confidence based on finger shape (should be elongated)
        const fingerWidth = this.calculateFingerWidth(largestComponent, fingerTip, fingerBase);
        const aspectRatio = fingerLength / Math.max(fingerWidth, 1);
        
        // More strict confidence calculation
        let confidence = 0;
        if (aspectRatio > 2.5) {
            confidence = Math.min((aspectRatio - 2.5) / 2.5, 1);
        }

        // Additional validation: check if tip is actually at the top
        const tipY = fingerTip.y;
        const baseY = fingerBase.y;
        if (tipY >= baseY) {
            confidence *= 0.5; // Reduce confidence if tip is not above base
        }

        // Only return if confidence is reasonable
        if (confidence < 0.3) return null;

        return {
            tip: { x: fingerTip.x, y: fingerTip.y },
            base: { x: fingerBase.x, y: fingerBase.y },
            angle: angle,
            confidence: confidence
        };
    }

    // Flood fill algorithm to find connected components
    floodFill(skinMask, visited, startX, startY, width, height) {
        const stack = [{ x: startX, y: startY }];
        const component = [];

        while (stack.length > 0) {
            const { x, y } = stack.pop();
            const index = y * width + x;

            if (x < 0 || x >= width || y < 0 || y >= height || 
                visited[index] || skinMask[index] !== 1) {
                continue;
            }

            visited[index] = true;
            component.push({ x, y });

            // Add 8-connected neighbors
            for (let dx = -1; dx <= 1; dx++) {
                for (let dy = -1; dy <= 1; dy++) {
                    if (dx === 0 && dy === 0) continue;
                    stack.push({ x: x + dx, y: y + dy });
                }
            }
        }

        return component;
    }

    // Calculate approximate finger width
    calculateFingerWidth(component, tip, base) {
        const midX = (tip.x + base.x) / 2;
        const midY = (tip.y + base.y) / 2;
        
        // Find points perpendicular to finger direction
        const perpAngle = Math.atan2(tip.y - base.y, tip.x - base.x) + Math.PI / 2;
        const perpX = Math.cos(perpAngle);
        const perpY = Math.sin(perpAngle);

        let maxWidth = 0;
        for (const point of component) {
            const dist = Math.abs((point.x - midX) * perpX + (point.y - midY) * perpY);
            maxWidth = Math.max(maxWidth, dist);
        }

        return maxWidth * 2; // Approximate width
    }

    // Advanced hand detection using color analysis
    detectHandByColor() {
        if (!this.video || !this.canvas) return { x: this.canvas.width / 2, y: this.canvas.height / 2 };

        // Create a temporary canvas to analyze video frame
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = this.video.videoWidth || this.video.clientWidth;
        tempCanvas.height = this.video.videoHeight || this.video.clientHeight;

        // Draw current video frame
        tempCtx.drawImage(this.video, 0, 0, tempCanvas.width, tempCanvas.height);

        // Get image data
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const data = imageData.data;

        // Look for skin-colored pixels
        let skinPixels = [];
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            
            // Simple skin color detection
            if (this.isSkinColor(r, g, b)) {
                const pixelIndex = i / 4;
                const x = pixelIndex % tempCanvas.width;
                const y = Math.floor(pixelIndex / tempCanvas.width);
                skinPixels.push({ x, y });
            }
        }

        // Calculate center of skin pixels
        if (skinPixels.length > 0) {
            const centerX = skinPixels.reduce((sum, p) => sum + p.x, 0) / skinPixels.length;
            const centerY = skinPixels.reduce((sum, p) => sum + p.y, 0) / skinPixels.length;
            return { x: centerX, y: centerY };
        }

        return { x: this.canvas.width / 2, y: this.canvas.height / 2 };
    }

    isSkinColor(r, g, b) {
        // Improved skin color detection with multiple conditions
        const total = r + g + b;
        if (total < 50) return false; // Too dark
        if (total > 750) return false; // Too bright

        const rRatio = r / total;
        const gRatio = g / total;
        const bRatio = b / total;

        // Primary skin color check
        const isSkinLike = rRatio > 0.3 && rRatio < 0.7 && 
                          gRatio > 0.2 && gRatio < 0.6 && 
                          bRatio > 0.1 && bRatio < 0.5;

        // Additional check: red should be dominant
        const redDominant = r > g && r > b;
        
        // Additional check: not too blue
        const notTooBlue = b < (r + g) / 2;

        // Additional check: reasonable brightness
        const reasonableBrightness = total > 100 && total < 600;

        return isSkinLike && redDominant && notTooBlue && reasonableBrightness;
    }

    // Check if position is stable (not a sudden jump)
    isPositionStable(newX, newY) {
        if (this.positionHistory.length === 0) return true;
        
        const lastPos = this.positionHistory[this.positionHistory.length - 1];
        const distance = Math.sqrt(
            Math.pow(newX - lastPos.x, 2) + Math.pow(newY - lastPos.y, 2)
        );
        
        // Reject if movement is too large (likely a false detection)
        return distance < 50;
    }

    // Get average position from history for stability
    getAveragePosition() {
        if (this.positionHistory.length === 0) {
            return { x: this.handPosition.x, y: this.handPosition.y };
        }
        
        const sumX = this.positionHistory.reduce((sum, pos) => sum + pos.x, 0);
        const sumY = this.positionHistory.reduce((sum, pos) => sum + pos.y, 0);
        
        return {
            x: sumX / this.positionHistory.length,
            y: sumY / this.positionHistory.length
        };
    }

    // Show tracking warning
    showTrackingWarning() {
        const warningElement = document.getElementById('trackingWarning');
        if (warningElement) {
            warningElement.style.display = 'block';
            // Hide warning after 5 seconds
            setTimeout(() => {
                warningElement.style.display = 'none';
            }, 5000);
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HandTracker;
} else {
    window.HandTracker = HandTracker;
}
