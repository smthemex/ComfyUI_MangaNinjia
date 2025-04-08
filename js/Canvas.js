
export class Canvas {
    constructor(node, widget) {
        this.node = node;
        this.widget = widget;
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.width = 512;
        this.height = 512;
        this.layers = [];
        this.selectedLayer = null;
        this.isRotating = false;
        this.rotationStartAngle = 0;
        this.rotationCenter = { x: 0, y: 0 };
        this.selectedLayers = [];
        this.isCtrlPressed = false;
        this.clickedPoints = [] // 用于存储点击的点
        this.allClickedPointsUrl = []; 
        this.offscreenCanvas = document.createElement('canvas');
        this.offscreenCtx = this.offscreenCanvas.getContext('2d', {
            alpha: false
        });
        this.gridCache = document.createElement('canvas');
        this.gridCacheCtx = this.gridCache.getContext('2d', {
            alpha: false
        });
        
        this.renderAnimationFrame = null;
        this.lastRenderTime = 0;
        this.renderInterval = 1000 / 60;
        this.isDirty = false;
        
        this.dataInitialized = false;
        this.pendingDataCheck = null;
        this.mouseX = 0; // 当前鼠标的 X 坐标
        this.mouseY = 0; // 当前鼠标的 Y 坐标
        this.initCanvas();
        this.setupEventListeners();
        this.initNodeData();
        
        // 添加混合模式列表
        this.blendModes = [
            { name: 'normal', label: '正常' },
            { name: 'multiply', label: '正片叠底' },
            { name: 'screen', label: '滤色' },
            { name: 'overlay', label: '叠加' },
            { name: 'darken', label: '变暗' },
            { name: 'lighten', label: '变亮' },
            { name: 'color-dodge', label: '颜色减淡' },
            { name: 'color-burn', label: '颜色加深' },
            { name: 'hard-light', label: '强光' },
            { name: 'soft-light', label: '柔光' },
            { name: 'difference', label: '差值' },
            { name: 'exclusion', label: '排除' }
        ];
        
        this.selectedBlendMode = null;
        this.blendOpacity = 100;
        this.isAdjustingOpacity = false;
        
        // 添加不透明度属性
        this.layers = this.layers.map(layer => ({
            ...layer,
            opacity: 1 // 默认不透明度为 1
        }));
    }

    /**
     * 更新画布尺寸
     * @param {number} width - 新的画布宽度
     * @param {number} height - 新的画布高度
     */
    updateCanvasSize(width, height) {
        // 更新画布尺寸
        this.width = width;
        this.height = height;
        this.canvas.width = width;
        this.canvas.height = height;

        // 调整所有图层的位置和大小
        this.layers.forEach(layer => {
            // 计算缩放比例
            const scale = Math.min(
                width / layer.image.width * 0.8, // 保留 80% 的边距
                height / layer.image.height * 0.8
            );

            // 更新图层尺寸
            layer.width = layer.image.width * scale;
            layer.height = layer.image.height * scale;

            // 更新图层位置，使其居中
            layer.x = (width - layer.width) / 2;
            layer.y = (height - layer.height) / 2;
        });

        // 重新渲染画布
        this.render();
    }



    initCanvas() {
        this.canvas.width = this.width;
        this.canvas.height = this.height;
        this.canvas.style.border = '1px solid black';
        this.canvas.style.maxWidth = '100%';
        this.canvas.style.backgroundColor = '#606060';
    }

    setupEventListeners() {
        let isDragging = false;
        let lastX = 0;
        let lastY = 0;
        let isRotating = false;
        let isResizing = false;
        let resizeHandle = null;
        let lastClickTime = 0;
        let isAltPressed = false;
        let dragStartX = 0;
        let dragStartY = 0;
        let originalWidth = 0;
        let originalHeight = 0;

        let drawnSquares = []; // 用于存储绘制的方块
        let squareCounter = 0; // 用于计数的变量
        // 其他事件监听器代码...
    
        // 添加 mousemove 事件监听器
        
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this.mouseX = -1;
            this.mouseY = -1;
        });

        // 添加点击事件监听器
        this.canvas.addEventListener('mousedown', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this.mouseX = e.clientX - rect.left; // 更新鼠标的 X 坐标
            this.mouseY = e.clientY - rect.top;  // 更新鼠标的 Y 坐标

        });
    

        // 绘制随机颜色方块
        const drawRandomSquare = (x, y) => {
            // const randomColor = `rgb(${Math.floor(Math.random() * 256)}, ${Math.floor(Math.random() * 256)}, ${Math.floor(Math.random() * 256)})`;
            const randomColor = '#00BFFF'; // 亮蓝色
            const square = {
                x: x - 5, // 使方块中心在点击位置
                y: y - 5,
                width: 10,
                height: 10,
                color: randomColor,
                number: squareCounter // 添加数字属性
            };
            drawnSquares.push(square);
            squareCounter++; // 增加计数
            this.render();
        };

        // 取消上一次绘制
        const undoLastSquare = () => {
            if (drawnSquares.length > 0) {
                drawnSquares.pop();// 移除绘制的最后一个方块
                this.clickedPoints.pop(); // 同步移除点击的最后一个坐标
                squareCounter--; // 减少计数
                this.render();
            }
        };

        // 取消所有绘制
        const clearAllSquares = () => {
            drawnSquares = []; // 清空所有方块
            this.clickedPoints = []; // 同步清空所有点击的坐标
            squareCounter = 0; // 重置计数
            this.render();
        };

        // 鼠标点击事件处理
        const handleMouseClick = (e) => {
            e.preventDefault(); // 阻止默认行为（如右键菜单）
            e.stopPropagation(); // 阻止事件冒泡

            console.log("Mouse click event triggered"); // 调试日志

            const rect = this.canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left; // 鼠标点击的原始X坐标
            const mouseY = e.clientY - rect.top;  // 鼠标点击的原始Y坐标

            // 计算缩放比例
            const scaleX = this.width / rect.width;  // 画布宽度缩放比例
            const scaleY = this.height / rect.height; // 画布高度缩放比例
            
            // 将鼠标点击的坐标映射到缩放后的画布坐标
            const x = mouseX * scaleX;
            const y = mouseY * scaleY;

            

            if (e.button === 0) { // 左键
                if (this.selectedLayer) {
                    const layer = this.selectedLayer;
                    const originalX = (x - layer.x) / (layer.width / layer.image.width);
                    const originalY = (y - layer.y) / (layer.height / layer.image.height);
        
                    // 将原始坐标添加到列表中
                    // this.clickedPoints.push({ x: originalX, y: originalY });
                    // 将原始坐标添加到列表中，并附带计数
                    this.clickedPoints.push({ 
                        x: originalX, 
                        y: originalY, 
                        number: squareCounter // 添加计数
                    });
                    console.log("Clicked point added:", { x: originalX, y: originalY });
                }
        
                drawRandomSquare(x, y);
            } else if (e.button === 2) { // 右键
                undoLastSquare();
            } else if (e.button === 1) { // 中键
                clearAllSquares();
            }
        };

        // 添加鼠标点击事件监听器
        this.canvas.addEventListener('mousedown', handleMouseClick);

        // 其他事件监听器代码...
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Control') {
                this.isCtrlPressed = true;
            }
            if (e.key === 'Alt') {
                isAltPressed = true;
                e.preventDefault();
            }
            if (e.key === 'Delete' && this.selectedLayer) {
                const index = this.layers.indexOf(this.selectedLayer);
                this.removeLayer(index);
            }
        });

        document.addEventListener('keyup', (e) => {
            if (e.key === 'Control') {
                this.isCtrlPressed = false;
            }
            if (e.key === 'Alt') {
                isAltPressed = false;
            }
        });

        // 在渲染逻辑中绘制方块
        this.actualRender = () => {
            if (this.offscreenCanvas.width !== this.width || 
                this.offscreenCanvas.height !== this.height) {
                this.offscreenCanvas.width = this.width;
                this.offscreenCanvas.height = this.height;
            }

            const ctx = this.offscreenCtx;
            
            ctx.fillStyle = '#606060';
            ctx.fillRect(0, 0, this.width, this.height);
            
            this.drawCachedGrid();
            
            const sortedLayers = [...this.layers].sort((a, b) => a.zIndex - b.zIndex);
            
            sortedLayers.forEach(layer => {
                if (!layer.image) return;
                
                ctx.save();
                
                // 应用混合模式和不透明度
                ctx.globalCompositeOperation = layer.blendMode || 'normal';
                ctx.globalAlpha = layer.opacity !== undefined ? layer.opacity : 1;
                
                const centerX = layer.x + layer.width/2;
                const centerY = layer.y + layer.height/2;
                const rad = layer.rotation * Math.PI / 180;
                
                // 1. 先设置变换
                ctx.setTransform(
                    Math.cos(rad), Math.sin(rad),
                    -Math.sin(rad), Math.cos(rad),
                    centerX, centerY
                );
                
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = 'high';
                
                // 2. 先绘制原始图像
                ctx.drawImage(
                    layer.image,
                    -layer.width/2,
                    -layer.height/2,
                    layer.width,
                    layer.height
                );
                
                // 3. 再应用遮罩
                if (layer.mask) {
                    try {
                        console.log("Applying mask to layer");
                        const maskCanvas = document.createElement('canvas');
                        const maskCtx = maskCanvas.getContext('2d');
                        maskCanvas.width = layer.width;
                        maskCanvas.height = layer.height;
                        
                        const maskImageData = maskCtx.createImageData(layer.width, layer.height);
                        const maskData = new Float32Array(layer.mask);
                        for (let i = 0; i < maskData.length; i++) {
                            maskImageData.data[i * 4] = 
                            maskImageData.data[i * 4 + 1] = 
                            maskImageData.data[i * 4 + 2] = 255;
                            maskImageData.data[i * 4 + 3] = maskData[i] * 255;
                        }
                        maskCtx.putImageData(maskImageData, 0, 0);
                        
                        // 使用destination-in混合模式
                        ctx.globalCompositeOperation = 'destination-in';
                        ctx.drawImage(maskCanvas, 
                            -layer.width/2, -layer.height/2,
                            layer.width, layer.height
                        );
                        
                        console.log("Mask applied successfully");
                    } catch (error) {
                        console.error("Error applying mask:", error);
                    }
                }
                
                // 4. 最后绘制选择框
                if (this.selectedLayers.includes(layer)) {
                    this.drawSelectionFrame(layer);
                }
                
                ctx.restore();
            });

            // 绘制所有方块
            drawnSquares.forEach(square => {
                // 绘制方块背景
                ctx.fillStyle = square.color;
                ctx.fillRect(square.x, square.y, square.width, square.height);
            // 绘制数字
            ctx.fillStyle = '#000'; // 数字颜色
            ctx.font = '8px Arial'; // 字体大小
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(square.number.toString(), square.x + square.width / 2, square.y + square.height / 2);
            });
            
            this.ctx.drawImage(this.offscreenCanvas, 0, 0);
        };
    }
        


    addLayer(image) {
        try {
            console.log("Adding layer with image:", image);
            
            const layer = {
                image: image,
                x: (this.width - image.width) / 2,
                y: (this.height - image.height) / 2,
                width: image.width,
                height: image.height,
                rotation: 0,
                zIndex: this.layers.length,
                blendMode: 'normal',  // 添加默认混合模式
                opacity: 1  // 添加默认透明度
            };
            
            this.layers.push(layer);
            this.selectedLayer = layer;
            this.render();
            
            console.log("Layer added successfully");
        } catch (error) {
            console.error("Error adding layer:", error);
            throw error;
        }
    }

    removeLayer(index) {
        if (index >= 0 && index < this.layers.length) {
            this.layers.splice(index, 1);
            this.selectedLayer = this.layers[this.layers.length - 1] || null;
            this.render();
        }
    }

    moveLayer(fromIndex, toIndex) {
        if (fromIndex >= 0 && fromIndex < this.layers.length &&
            toIndex >= 0 && toIndex < this.layers.length) {
            const layer = this.layers.splice(fromIndex, 1)[0];
            this.layers.splice(toIndex, 0, layer);
            this.render();
        }
    }



    updateCanvasSize(width, height) {
        this.width = width;
        this.height = height;
        
        this.canvas.width = width;
        this.canvas.height = height;
        
        // 调整所有图层的位置和大小
        this.layers.forEach(layer => {
            const scale = Math.min(
                width / layer.image.width * 0.8,
                height / layer.image.height * 0.8
            );
            layer.width = layer.image.width * scale;
            layer.height = layer.image.height * scale;
            layer.x = (width - layer.width) / 2;
            layer.y = (height - layer.height) / 2;
        });
        
        this.render();
    }

    render() {
        if (this.renderAnimationFrame) {
            this.isDirty = true;
            return;
        }
        
        this.renderAnimationFrame = requestAnimationFrame(() => {
            const now = performance.now();
            if (now - this.lastRenderTime >= this.renderInterval) {
                this.lastRenderTime = now;
                this.actualRender();
                this.isDirty = false;
            }
            
            if (this.isDirty) {
                this.renderAnimationFrame = null;
                this.render();
            } else {
                this.renderAnimationFrame = null;
            }
        });
    }

    actualRender() {
        if (this.offscreenCanvas.width !== this.width || 
            this.offscreenCanvas.height !== this.height) {
            this.offscreenCanvas.width = this.width;
            this.offscreenCanvas.height = this.height;
        }

        const ctx = this.offscreenCtx;
        
        ctx.fillStyle = '#606060';
        ctx.fillRect(0, 0, this.width, this.height);
        
        this.drawCachedGrid();
        
        const sortedLayers = [...this.layers].sort((a, b) => a.zIndex - b.zIndex);
        
        sortedLayers.forEach(layer => {
            if (!layer.image) return;
            
            ctx.save();
            
            // 应用混合模式和不透明度
            ctx.globalCompositeOperation = layer.blendMode || 'normal';
            ctx.globalAlpha = layer.opacity !== undefined ? layer.opacity : 1;
            
            const centerX = layer.x + layer.width/2;
            const centerY = layer.y + layer.height/2;
            const rad = layer.rotation * Math.PI / 180;
            
            // 1. 先设置变换
            ctx.setTransform(
                Math.cos(rad), Math.sin(rad),
                -Math.sin(rad), Math.cos(rad),
                centerX, centerY
            );
            
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            
            // 2. 先绘制原始图像
            ctx.drawImage(
                layer.image,
                -layer.width/2,
                -layer.height/2,
                layer.width,
                layer.height
            );
            
            // 3. 再应用遮罩
            if (layer.mask) {
                try {
                    console.log("Applying mask to layer");
                    const maskCanvas = document.createElement('canvas');
                    const maskCtx = maskCanvas.getContext('2d');
                    maskCanvas.width = layer.width;
                    maskCanvas.height = layer.height;
                    
                    const maskImageData = maskCtx.createImageData(layer.width, layer.height);
                    const maskData = new Float32Array(layer.mask);
                    for (let i = 0; i < maskData.length; i++) {
                        maskImageData.data[i * 4] = 
                        maskImageData.data[i * 4 + 1] = 
                        maskImageData.data[i * 4 + 2] = 255;
                        maskImageData.data[i * 4 + 3] = maskData[i] * 255;
                    }
                    maskCtx.putImageData(maskImageData, 0, 0);
                    
                    // 使用destination-in混合模式
                    ctx.globalCompositeOperation = 'destination-in';
                    ctx.drawImage(maskCanvas, 
                        -layer.width/2, -layer.height/2,
                        layer.width, layer.height
                    );
                    
                    console.log("Mask applied successfully");
                } catch (error) {
                    console.error("Error applying mask:", error);
                }
            }
            
            // 4. 最后绘制选择框
            if (this.selectedLayers.includes(layer)) {
                this.drawSelectionFrame(layer);
            }
            
            ctx.restore();
        });
        
        this.ctx.drawImage(this.offscreenCanvas, 0, 0);
    }

    /**
     * 获取所有点击的原始坐标列表
     * @returns {Array<{x: number, y: number}>} 点击的原始坐标列表
     */
    getClickedPoints() {
        return this.clickedPoints;
    }

    drawCachedGrid() {
        if (this.gridCache.width !== this.width || 
            this.gridCache.height !== this.height) {
            this.gridCache.width = this.width;
            this.gridCache.height = this.height;
            
            const ctx = this.gridCacheCtx;
            const gridSize = 20;
            
            ctx.beginPath();
            ctx.strokeStyle = '#e0e0e0';
            ctx.lineWidth = 0.5;
            
            for(let y = 0; y < this.height; y += gridSize) {
                ctx.moveTo(0, y);
                ctx.lineTo(this.width, y);
            }
            
            for(let x = 0; x < this.width; x += gridSize) {
                ctx.moveTo(x, 0);
                ctx.lineTo(x, this.height);
            }
            
            ctx.stroke();
        }
        
        this.offscreenCtx.drawImage(this.gridCache, 0, 0);
    }

    drawSelectionFrame(layer) {
        const ctx = this.offscreenCtx;
        
        ctx.beginPath();
        
        ctx.rect(-layer.width/2, -layer.height/2, layer.width, layer.height);
        
        ctx.moveTo(0, -layer.height/2);
        ctx.lineTo(0, -layer.height/2 - 20);
        
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        ctx.beginPath();
        
        const points = [
            {x: 0, y: -layer.height/2 - 20},
            {x: -layer.width/2, y: -layer.height/2},
            {x: layer.width/2, y: -layer.height/2},
            {x: layer.width/2, y: layer.height/2},
            {x: -layer.width/2, y: layer.height/2}
        ];
        
        points.forEach(point => {
            ctx.moveTo(point.x, point.y);
            ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
        });
        
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.stroke();
    }
    async saveToServer(fileName,ifdownload=false) {
        return new Promise((resolve) => {
            // 创建临时画布
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = this.width;
            tempCanvas.height = this.height;
            const tempCtx = tempCanvas.getContext('2d');

            // 填充白色背景
            tempCtx.fillStyle = '#ffffff';
            tempCtx.fillRect(0, 0, this.width, this.height);

            // 按照 zIndex 顺序绘制所有图层
            this.layers.sort((a, b) => a.zIndex - b.zIndex).forEach(layer => {
                tempCtx.save();
                tempCtx.globalCompositeOperation = layer.blendMode || 'normal';
                tempCtx.globalAlpha = layer.opacity !== undefined ? layer.opacity : 1;
                tempCtx.translate(layer.x + layer.width / 2, layer.y + layer.height / 2);
                tempCtx.rotate(layer.rotation * Math.PI / 180);
                tempCtx.drawImage(
                    layer.image,
                    -layer.width / 2,
                    -layer.height / 2,
                    layer.width,
                    layer.height
                );
                tempCtx.restore();
            });

            
            // 将 canvas 内容转换为 PNG 数据流
            tempCanvas.toBlob(async (blob) => {
                const formData = new FormData();
                formData.append("image", blob, fileName);
                formData.append("overwrite", "true");
        
                try {
                    const resp = await fetch("/upload/image", {
                        method: "POST",
                        body: formData,
                    });

                    if (resp.status === 200) {
                        this.saveClickedPointsAsJson(fileName, ifdownload);
                        resolve(true);
                    } else {
                        console.error("保存失败:", resp.status);
                        resolve(false);
                    }
                } catch (error) {
                    console.error("保存错误:", error);
                    resolve(false);
                }
            }, "image/png");
        });
        
    }
 
    

    saveClickedPointsAsJson(fileName, shouldDownload = false) {
        console.log("当前鼠标位置:", { mouseX: this.mouseX, mouseY: this.mouseY });


        // 如果点击位置不在画布范围内，直接返回
        const isOutsideCanvas =(this.mouseX < 0 || this.mouseX > this.width || this.mouseY < 0 || this.mouseY > this.height)
        if (isOutsideCanvas ) {
            console.log("点击位置不在画布范围内，忽略操作");
            return;
        }

        // 替换文件名中的 .png 为 .json
        const jsonFileName = fileName.replace('.png', '.json');
        const clickedPoints = this.getClickedPoints();
         // 去重逻辑
        const uniqueClickedPoints = [];
        const seenPoints = new Set();

        for (const point of clickedPoints) {
            const key = `${point.x},${point.y}`; // 使用 x 和 y 组合作为唯一标识
            if (!seenPoints.has(key)) {
                seenPoints.add(key);
                uniqueClickedPoints.push(point); // 保留原始点数据
            }
        }

        const jsonContent = JSON.stringify(uniqueClickedPoints, null, 2);
    
        const blob = new Blob([jsonContent], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        this.allClickedPointsUrl.push(url);

        if (shouldDownload && !isOutsideCanvas ) {
            const a = document.createElement('a');
            a.href = url;
            a.download = jsonFileName;
            document.body.appendChild(a);
            a.click();
    
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

        }}


    moveLayerUp() {
        if (!this.selectedLayer) return;
        const index = this.layers.indexOf(this.selectedLayer);
        if (index < this.layers.length - 1) {
            const temp = this.layers[index].zIndex;
            this.layers[index].zIndex = this.layers[index + 1].zIndex;
            this.layers[index + 1].zIndex = temp;
            [this.layers[index], this.layers[index + 1]] = [this.layers[index + 1], this.layers[index]];
            this.render();
        }
    }

    moveLayerDown() {
        if (!this.selectedLayer) return;
        const index = this.layers.indexOf(this.selectedLayer);
        if (index > 0) {
            const temp = this.layers[index].zIndex;
            this.layers[index].zIndex = this.layers[index - 1].zIndex;
            this.layers[index - 1].zIndex = temp;
            [this.layers[index], this.layers[index - 1]] = [this.layers[index - 1], this.layers[index]];
            this.render();
        }
    }

    getLayerAtPosition(x, y) {
        // 获取画布的实际显示尺寸和位置
        const rect = this.canvas.getBoundingClientRect();
        
        // 计算画布的缩放比例
        const displayWidth = rect.width;
        const displayHeight = rect.height;
        const scaleX = this.width / displayWidth;
        const scaleY = this.height / displayHeight;
        
        // 计算鼠标在画布上的实际位置
        const canvasX = (x) * scaleX;
        const canvasY = (y) * scaleY;
        
        // 从上层到下层遍历所有图层
        for (let i = this.layers.length - 1; i >= 0; i--) {
            const layer = this.layers[i];
            
            // 计算旋转后的点击位置
            const centerX = layer.x + layer.width/2;
            const centerY = layer.y + layer.height/2;
            const rad = -layer.rotation * Math.PI / 180;
            
            // 将点击坐标转换到图层的本地坐标系
            const dx = canvasX - centerX;
            const dy = canvasY - centerY;
            const rotatedX = dx * Math.cos(rad) - dy * Math.sin(rad) + centerX;
            const rotatedY = dx * Math.sin(rad) + dy * Math.cos(rad) + centerY;
            
            // 检查点击位置是否在图层范围内
            if (rotatedX >= layer.x && 
                rotatedX <= layer.x + layer.width &&
                rotatedY >= layer.y && 
                rotatedY <= layer.y + layer.height) {
                
                // 创建临时画布来检查透明度
                const tempCanvas = document.createElement('canvas');
                const tempCtx = tempCanvas.getContext('2d');
                tempCanvas.width = layer.width;
                tempCanvas.height = layer.height;
                
                // 绘制图层到临时画布
                tempCtx.save();
                tempCtx.clearRect(0, 0, layer.width, layer.height);
                tempCtx.drawImage(
                    layer.image,
                    0,
                    0,
                    layer.width,
                    layer.height
                );
                tempCtx.restore();
                
                // 获取点击位置的像素数据
                const localX = rotatedX - layer.x;
                const localY = rotatedY - layer.y;
                
                try {
                    const pixel = tempCtx.getImageData(
                        Math.round(localX), 
                        Math.round(localY), 
                        1, 1
                    ).data;
                    // 检查像素的alpha值
                    if (pixel[3] > 10) {
                        return {
                            layer: layer,
                            localX: localX,
                            localY: localY
                        };
                    }
                } catch(e) {
                    console.error("Error checking pixel transparency:", e);
                }
            }
        }
        return null;
    }

    getResizeHandle(x, y) {
        if (!this.selectedLayer) return null;
        
        const handleRadius = 5;
        const handles = {
            'nw': {x: this.selectedLayer.x, y: this.selectedLayer.y},
            'ne': {x: this.selectedLayer.x + this.selectedLayer.width, y: this.selectedLayer.y},
            'se': {x: this.selectedLayer.x + this.selectedLayer.width, y: this.selectedLayer.y + this.selectedLayer.height},
            'sw': {x: this.selectedLayer.x, y: this.selectedLayer.y + this.selectedLayer.height}
        };

        for (const [position, point] of Object.entries(handles)) {
            if (Math.sqrt(Math.pow(x - point.x, 2) + Math.pow(y - point.y, 2)) <= handleRadius) {
                return position;
            }
        }
        return null;
    }

    // 修改水平镜像方法
    mirrorHorizontal() {
        if (!this.selectedLayer) return;
        
        // 创建临时画布
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = this.selectedLayer.image.width;
        tempCanvas.height = this.selectedLayer.image.height;
        
        // 水平翻转绘制
        tempCtx.translate(tempCanvas.width, 0);
        tempCtx.scale(-1, 1);
        tempCtx.drawImage(this.selectedLayer.image, 0, 0);
        
        // 创建新图像
        const newImage = new Image();
        newImage.onload = () => {
            this.selectedLayer.image = newImage;
            this.render();
        };
        newImage.src = tempCanvas.toDataURL();
    }

    // 修改垂直镜像方法
    mirrorVertical() {
        if (!this.selectedLayer) return;
        
        // 创建临时画布
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = this.selectedLayer.image.width;
        tempCanvas.height = this.selectedLayer.image.height;
        
        // 垂直翻转绘制
        tempCtx.translate(0, tempCanvas.height);
        tempCtx.scale(1, -1);
        tempCtx.drawImage(this.selectedLayer.image, 0, 0);
        
        // 创建新图像
        const newImage = new Image();
        newImage.onload = () => {
            this.selectedLayer.image = newImage;
            this.render();
        };
        newImage.src = tempCanvas.toDataURL();
    }

    async getLayerImageData(layer) {
        try {
            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');
            
            // 设置画布尺寸
            tempCanvas.width = layer.width;
            tempCanvas.height = layer.height;
            
            // 清除画布
            tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
            
            // 绘制图层
            tempCtx.save();
            tempCtx.translate(layer.width/2, layer.height/2);
            tempCtx.rotate(layer.rotation * Math.PI / 180);
            tempCtx.drawImage(
                layer.image,
                -layer.width/2,
                -layer.height/2,
                layer.width,
                layer.height
            );
            tempCtx.restore();
            
            // 获取base64数据
            const dataUrl = tempCanvas.toDataURL('image/png');
            if (!dataUrl.startsWith('data:image/png;base64,')) {
                throw new Error("Invalid image data format");
            }
            
            return dataUrl;
        } catch (error) {
            console.error("Error getting layer image data:", error);
            throw error;
        }
    }

    // 添加带遮罩的图层
    addMattedLayer(image, mask) {
        const layer = {
            image: image,
            mask: mask,
            x: 0,
            y: 0,
            width: image.width,
            height: image.height,
            rotation: 0,
            zIndex: this.layers.length
        };
        
        this.layers.push(layer);
        this.selectedLayer = layer;
        this.render();
    }


    async addInputToCanvas(inputImage, inputMask) {
        try {
            console.log("Adding input to canvas:", { inputImage });
            
            // 创建临时画布
            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');
            tempCanvas.width = inputImage.width;
            tempCanvas.height = inputImage.height;

            // 将数据绘制到临时画布
            const imgData = new ImageData(
                inputImage.data,
                inputImage.width,
                inputImage.height
            );
            tempCtx.putImageData(imgData, 0, 0);

            // 创建新图像
            const image = new Image();
            await new Promise((resolve, reject) => {
                image.onload = resolve;
                image.onerror = reject;
                image.src = tempCanvas.toDataURL();
            });

            // 计算缩放比例
            const scale = Math.min(
                this.width / inputImage.width * 0.8,
                this.height / inputImage.height * 0.8
            );

            // 创建新图层
            const layer = {
                image: image,
                x: (this.width - inputImage.width * scale) / 2,
                y: (this.height - inputImage.height * scale) / 2,
                width: inputImage.width * scale,
                height: inputImage.height * scale,
                rotation: 0,
                zIndex: this.layers.length
            };

            // 如果有遮罩数据，添加到图层
            if (inputMask) {
                layer.mask = inputMask.data;
            }

            // 添加图层并选中
            this.layers.push(layer);
            this.selectedLayer = layer;
            
            // 渲染画布
            this.render();
            console.log("Layer added successfully");
            
            return true;

        } catch (error) {
            console.error("Error in addInputToCanvas:", error);
            throw error;
        }
    }

    // 改进图像转换方法
    async convertTensorToImage(tensor) {
        try {
            console.log("Converting tensor to image:", tensor);
            
            if (!tensor || !tensor.data || !tensor.width || !tensor.height) {
                throw new Error("Invalid tensor data");
            }

            // 创建临时画布
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = tensor.width;
            canvas.height = tensor.height;

            // 创建像数据
            const imageData = new ImageData(
                new Uint8ClampedArray(tensor.data),
                tensor.width,
                tensor.height
            );

            // 将数据绘制到画布
            ctx.putImageData(imageData, 0, 0);

            // 创建新图像
            return new Promise((resolve, reject) => {
                const img = new Image();
                img.onload = () => resolve(img);
                img.onerror = (e) => reject(new Error("Failed to load image: " + e));
                img.src = canvas.toDataURL();
            });
        } catch (error) {
            console.error("Error converting tensor to image:", error);
            throw error;
        }
    }

    // 改进遮罩转换方法
    async convertTensorToMask(tensor) {
        if (!tensor || !tensor.data) {
            throw new Error("Invalid mask tensor");
        }

        try {
            // 确保数据是Float32Array
            return new Float32Array(tensor.data);
        } catch (error) {
            throw new Error(`Mask conversion failed: ${error.message}`);
        }
    }

    // 改进数据初始化方法
    async initNodeData() {
        try {
            console.log("Starting node data initialization...");
            
            // 检查节点和输入是否存在
            if (!this.node || !this.node.inputs) {
                console.log("Node or inputs not ready");
                return this.scheduleDataCheck();
            }

            // 检查图像��入
            if (this.node.inputs[0] && this.node.inputs[0].link) {
                const imageLinkId = this.node.inputs[0].link;
                const imageData = app.nodeOutputs[imageLinkId];
                
                if (imageData) {
                    console.log("Found image data:", imageData);
                    await this.processImageData(imageData);
                    this.dataInitialized = true;
                } else {
                    console.log("Image data not available yet");
                    return this.scheduleDataCheck();
                }
            }

            // 检查遮罩输入
            if (this.node.inputs[1] && this.node.inputs[1].link) {
                const maskLinkId = this.node.inputs[1].link;
                const maskData = app.nodeOutputs[maskLinkId];
                
                if (maskData) {
                    console.log("Found mask data:", maskData);
                    await this.processMaskData(maskData);
                }
            }

        } catch (error) {
            console.error("Error in initNodeData:", error);
            return this.scheduleDataCheck();
        }
    }

    // 添加数据检查调度方法
    scheduleDataCheck() {
        if (this.pendingDataCheck) {
            clearTimeout(this.pendingDataCheck);
        }
        
        this.pendingDataCheck = setTimeout(() => {
            this.pendingDataCheck = null;
            if (!this.dataInitialized) {
                this.initNodeData();
            }
        }, 1000); // 1秒后重试
    }

    // 修改图像数据处理方法
    async processImageData(imageData) {
        try {
            if (!imageData) return;
            
            console.log("Processing image data:", {
                type: typeof imageData,
                isArray: Array.isArray(imageData),
                shape: imageData.shape,
                hasData: !!imageData.data
            });
            
            // 处理数组格式
            if (Array.isArray(imageData)) {
                imageData = imageData[0];
            }
            
            // 验证数据格式
            if (!imageData.shape || !imageData.data) {
                throw new Error("Invalid image data format");
            }
            
            // 保持原始尺寸和比例
            const originalWidth = imageData.shape[2];
            const originalHeight = imageData.shape[1];
            
            // 计算适当的缩放比例
            const scale = Math.min(
                this.width / originalWidth * 0.8,
                this.height / originalHeight * 0.8
            );
            
            // 转换数据
            const convertedData = this.convertTensorToImageData(imageData);
            if (convertedData) {
                const image = await this.createImageFromData(convertedData);
                
                // 使用计算的缩放比例添加图层
                this.addScaledLayer(image, scale);
                console.log("Image layer added successfully with scale:", scale);
            }
        } catch (error) {
            console.error("Error processing image data:", error);
            throw error;
        }
    }

    // 添加新的缩放图层方法
    addScaledLayer(image, scale) {
        try {
            const scaledWidth = image.width * scale;
            const scaledHeight = image.height * scale;
            
            const layer = {
                image: image,
                x: (this.width - scaledWidth) / 2,
                y: (this.height - scaledHeight) / 2,
                width: scaledWidth,
                height: scaledHeight,
                rotation: 0,
                zIndex: this.layers.length,
                originalWidth: image.width,
                originalHeight: image.height
            };
            
            this.layers.push(layer);
            this.selectedLayer = layer;
            this.render();
            
            console.log("Scaled layer added:", {
                originalSize: `${image.width}x${image.height}`,
                scaledSize: `${scaledWidth}x${scaledHeight}`,
                scale: scale
            });
        } catch (error) {
            console.error("Error adding scaled layer:", error);
            throw error;
        }
    }

    // 改进张量转换方法
    convertTensorToImageData(tensor) {
        try {
            const shape = tensor.shape;
            const height = shape[1];
            const width = shape[2];
            const channels = shape[3];
            
            console.log("Converting tensor:", {
                shape: shape,
                dataRange: {
                    min: tensor.min_val,
                    max: tensor.max_val
                }
            });
            
            // 创建图像数据
            const imageData = new ImageData(width, height);
            const data = new Uint8ClampedArray(width * height * 4);
            
            // 重建数据结构
            const flatData = tensor.data;
            const pixelCount = width * height;
            
            for (let i = 0; i < pixelCount; i++) {
                const pixelIndex = i * 4;
                const tensorIndex = i * channels;
                
                // 正确处理RGB通道
                for (let c = 0; c < channels; c++) {
                    const value = flatData[tensorIndex + c];
                    // 根据实际值范围行映射
                    const normalizedValue = (value - tensor.min_val) / (tensor.max_val - tensor.min_val);
                    data[pixelIndex + c] = Math.round(normalizedValue * 255);
                }
                
                // Alpha通道
                data[pixelIndex + 3] = 255;
            }
            
            imageData.data.set(data);
            return imageData;
        } catch (error) {
            console.error("Error converting tensor:", error);
            return null;
        }
    }

    // 添加图像创建方法
    async createImageFromData(imageData) {
        return new Promise((resolve, reject) => {
            const canvas = document.createElement('canvas');
            canvas.width = imageData.width;
            canvas.height = imageData.height;
            const ctx = canvas.getContext('2d');
            ctx.putImageData(imageData, 0, 0);

            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = canvas.toDataURL();
        });
    }

    // 添加数据重试机制
    async retryDataLoad(maxRetries = 3, delay = 1000) {
        for (let i = 0; i < maxRetries; i++) {
            try {
                await this.initNodeData();
                return;
            } catch (error) {
                console.warn(`Retry ${i + 1}/${maxRetries} failed:`, error);
                if (i < maxRetries - 1) {
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
        }
        console.error("Failed to load data after", maxRetries, "retries");
    }

    async processMaskData(maskData) {
        try {
            if (!maskData) return;
            
            console.log("Processing mask data:", maskData);
            
            // 处理数组格式
            if (Array.isArray(maskData)) {
                maskData = maskData[0];
            }
            
            // 检查数据格式
            if (!maskData.shape || !maskData.data) {
                throw new Error("Invalid mask data format");
            }
            
            // 如果有选中的图层，应用遮罩
            if (this.selectedLayer) {
                const maskTensor = await this.convertTensorToMask(maskData);
                this.selectedLayer.mask = maskTensor;
                this.render();
                console.log("Mask applied to selected layer");
            }
        } catch (error) {
            console.error("Error processing mask data:", error);
        }
    }
}
