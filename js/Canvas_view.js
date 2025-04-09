import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { $el } from "../../scripts/ui.js";
import { Canvas } from "./Canvas.js";



async function createCanvasWidget(node, widget, app) {
    // 创建一个新的Canvas对象
    const canvas = new Canvas(node, widget);

    // 添加全局样式
    const style = document.createElement('style');
    style.textContent = `
        .painter-button {
            background: linear-gradient(to bottom, #4a4a4a, #3a3a3a);
            border: 1px solid #2a2a2a;
            border-radius: 4px;
            color: #ffffff;
            padding: 6px 12px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
            min-width: 80px;
            text-align: center;
            margin: 2px;
            text-shadow: 0 1px 1px rgba(0,0,0,0.2);
        }

        .painter-button:hover {
            background: linear-gradient(to bottom, #5a5a5a, #4a4a4a);
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }

        .painter-button:active {
            background: linear-gradient(to bottom, #3a3a3a, #4a4a4a);
            transform: translateY(1px);
        }

        .painter-button.primary {
            background: linear-gradient(to bottom, #4a6cd4, #3a5cc4);
            border-color: #2a4cb4;
        }

        .painter-button.primary:hover {
            background: linear-gradient(to bottom, #5a7ce4, #4a6cd4);
        }

        .painter-controls {
            background: linear-gradient(to bottom, #404040, #383838);
            border-bottom: 1px solid #2a2a2a;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 8px;
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            align-items: center;
        }

        .painter-container {
            background: #607080;  /* 带蓝色的灰色背景 */
            border: 1px solid #4a5a6a;
            border-radius: 6px;
            box-shadow: inset 0 0 10px rgba(0,0,0,0.1);
        }

        .painter-dialog {
            background: #404040;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            padding: 20px;
            color: #ffffff;
        }

        .painter-dialog input {
            background: #303030;
            border: 1px solid #505050;
            border-radius: 4px;
            color: #ffffff;
            padding: 4px 8px;
            margin: 4px;
            width: 80px;
        }

        .painter-dialog button {
            background: #505050;
            border: 1px solid #606060;
            border-radius: 4px;
            color: #ffffff;
            padding: 4px 12px;
            margin: 4px;
            cursor: pointer;
        }

        .painter-dialog button:hover {
            background: #606060;
        }

        .blend-opacity-slider {
            width: 100%;
            margin: 5px 0;
            display: none;
        }
        
        .blend-mode-active .blend-opacity-slider {
            display: block;
        }
        
        .blend-mode-item {
            padding: 5px;
            cursor: pointer;
            position: relative;
        }
        
        .blend-mode-item.active {
            background-color: rgba(0,0,0,0.1);
        }
        /* 新增样式：点击后的颜色 */
        .painter-button.clicked {
            background: linear-gradient(to bottom,rgb(45, 114, 204),rgb(67, 119, 197));
        } 
    `;
    document.head.appendChild(style);

  
    // 监听图片输入事件
    const handleImageInput = async (file) => {
        const img = new Image();
        img.onload = () => {
            console.log("Image loaded:", img); // 调试信息
            canvas.updateCanvasSize(img.width, img.height);
    
            const layer = {
                image: img,
                x: (canvas.width - img.width) / 2,
                y: (canvas.height - img.height) / 2,
                width: img.width,
                height: img.height,
                rotation: 0,
                zIndex: canvas.layers.length
            };
            canvas.layers.push(layer);
            canvas.selectedLayer = layer;
    
            console.log("Layer added:", layer); // 调试信息
            canvas.render();
        };
        img.onerror = (e) => {
            console.error("Failed to load image:", e); // 错误处理
        };
        img.src = URL.createObjectURL(file);
    };

   
   
    // 修改控制面板，使其高度自适应
    const controlPanel = $el("div.painterControlPanel", {}, [
        $el("div.controls.painter-controls", {
            style: {
                position: "absolute",
                top: "0",
                left: "0",
                right: "0",
                minHeight: "50px", // 改为最小高度
                zIndex: "10",
                background: "linear-gradient(to bottom, #404040, #383838)",
                borderBottom: "1px solid #2a2a2a",
                boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
                padding: "8px",
                display: "flex",
                gap: "6px",
                flexWrap: "wrap",
                alignItems: "center"
            },
            // 添加监听器来动态整画布容器的位置
            onresize: (entries) => {
                const controlsHeight = entries[0].target.offsetHeight;
                canvasContainer.style.top = (controlsHeight + 10) + "px";
            }
        }, [
            $el("button.painter-button.primary", {
                textContent: "Add Image",
                onclick: () => {
                    const input = document.createElement('input');
                    input.type = 'file';
                    input.accept = 'image/*';
                    input.multiple = true;
                    input.onchange = async (e) => {
                        for (const file of e.target.files) {
                            handleImageInput(file);
                        }
                    };
                    input.click();
                }
            }),
            
            $el("button.painter-button", {
                textContent: "Remove Layer",
                onclick: (event) => {
                    const index = canvas.layers.indexOf(canvas.selectedLayer);
                    canvas.removeLayer(index);
                    event.target.classList.toggle('clicked');
                }
            }),
           
            $el("button.painter-button", {
                textContent: "Mirror H",
                onclick: (event) => {
                    canvas.mirrorHorizontal();
                    event.target.classList.toggle('clicked');
                }
            }),
            // 添加垂直镜像按钮
            $el("button.painter-button", {
                textContent: "Mirror V",
                onclick: (event) => {
                    canvas.mirrorVertical();
                    event.target.classList.toggle('clicked');
                }
            }),
       
        
        ])
    ]);


    // 创建ResizeObserver来监控控制面板的高度变化
    const resizeObserver = new ResizeObserver((entries) => {
        const controlsHeight = entries[0].target.offsetHeight;
        canvasContainer.style.top = (controlsHeight + 10) + "px";
    });

    // 监控控制面板的大小变化
    resizeObserver.observe(controlPanel.querySelector('.controls'));

    // 获取触发器widget
    const triggerWidget = node.widgets.find(w => w.name === "trigger");
    
    // 创建更新函数
    const updateOutput = async () => {
        // 保存画布
        await canvas.saveToServer(widget.value);
        // 更新触发器值
        triggerWidget.value = (triggerWidget.value + 1) % 99999999;
        // 触发节点更新
        app.graph.runStep();
    };

    // 修改所有可能触发更新的操作
    const addUpdateToButton = (button) => {
        const origClick = button.onclick;
        button.onclick = async (...args) => {
            await origClick?.(...args);
            await updateOutput();
        };
    };

    // 为所有按钮添加更新逻辑
    controlPanel.querySelectorAll('button').forEach(addUpdateToButton);

    // 修改画布容器样式，使用动态top值
    const canvasContainer = $el("div.painterCanvasContainer.painter-container", {
        style: {
            position: "absolute",
            top: "60px", // 初始值
            left: "10px",
            right: "10px",
            bottom: "10px",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            overflow: "hidden"
        }
    }, [canvas.canvas]);

    // 修改节点大小调整逻辑
    node.onResize = function() {
        const minSize = 300;
        const controlsElement = controlPanel.querySelector('.controls');
        const controlPanelHeight = controlsElement.offsetHeight; // 取实际高
        const padding = 20;
        
        // 保持节点宽度，高度根据画布比例调整
        const width = Math.max(this.size[0], minSize);
        const height = Math.max(
            width * (canvas.height / canvas.width) + controlPanelHeight + padding * 2,
            minSize + controlPanelHeight
        );
        
        this.size[0] = width;
        this.size[1] = height;
        
        // 计算画布的实际可用空间
        const availableWidth = width - padding * 2;
        const availableHeight = height - controlPanelHeight - padding * 2;
        
        // 更新画布尺寸，保持比例
        const scale = Math.min(
            availableWidth / canvas.width,
            availableHeight / canvas.height
        );
        
        canvas.canvas.style.width = (canvas.width * scale) + "px";
        canvas.canvas.style.height = (canvas.height * scale) + "px";
        
        // 强制重新渲染
        canvas.render();
    };

    // 添加点击事件监听
    canvas.canvas.addEventListener('mouseup', updateOutput);
    canvas.canvas.addEventListener('mouseleave', updateOutput);

    // 点击画布后，获取点击坐标列表
    const clickedPoints = canvas.getClickedPoints();
    console.log("Clicked points:", clickedPoints);

    // 创建一个包含控制面板和画布的容器
    const mainContainer = $el("div.painterMainContainer", {
        style: {
            position: "relative",
            width: "100%",
            height: "100%"
        }
    }, [controlPanel, canvasContainer]);

    // 将主容器添加到节点
    const mainWidget = node.addDOMWidget("mainContainer", "widget", mainContainer);

    // 设置节点的默认大小
    node.size = [512, 512]; // 设置初始大小为正方形

    // 在执行开始时保存数据
    api.addEventListener("execution_start", async () => {
        // 保存画布
        await canvas.saveToServer(widget.value);
        
        // 保存当前节点的输入数据
        // if (node.inputs[0].link) {
        //     const linkId = node.inputs[0].link;
        //     const inputData = app.nodeOutputs[linkId];
        //     if (inputData) {
        //         ImageCache.set(linkId, inputData);
        //     }
        // }
    });

    // 移除原来在 saveToServer 中的缓存清理
    const originalSaveToServer = canvas.saveToServer;
    canvas.saveToServer = async function(fileName) {
        const result = await originalSaveToServer.call(this, fileName);
        // 移除这里的缓存清理
        // ImageCache.clear();
        return result;
    };

    return {
        canvas: canvas,
        panel: controlPanel
    };
}



// 修改缓存管理
const ImageCache = {
    cache: new Map(),
    
    // 存储图像数据
    set(key, imageData) {
        console.log("Caching image data for key:", key);
        this.cache.set(key, imageData);
    },
    
    // 获取图像数据
    get(key) {
        const data = this.cache.get(key);
        console.log("Retrieved cached data for key:", key, !!data);
        return data;
    },
    
    // 检查是否存在
    has(key) {
        return this.cache.has(key);
    },
    
    // 清除缓存
    clear() {
        console.log("Clearing image cache");
        this.cache.clear();
    }
};



app.registerExtension({
    name: "Comfy.MarkImageNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "MarkImageNode") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = async function() {
                const r = onNodeCreated?.apply(this, arguments);
                
                const widget = this.widgets.find(w => w.name === "markimage_image");
                await createCanvasWidget(this, widget, app);
                
                return r;
            };
        }
    }
}); 
