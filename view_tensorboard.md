# Viewing TensorBoard Logs for wm_v5

## Steps

1. Activate the conda environment:
   ```bash
   conda activate verify2act
   ```

2. Start TensorBoard pointing at the wm_v5 log directory:
   ```bash
   tensorboard --logdir verify2act/output/wm_v5/tb_logs --port 6006
   ```

3. Open TensorBoard in a browser.

   **Option A — From bash (preferred), open in system browser:**
   ```bash
   xdg-open http://localhost:6006
   ```

   **Option B — From bash, open in VS Code's Simple Browser:**
   ```bash
   code --open-url "vscode://vscode.simpleBrowser.show?url=http%3A%2F%2Flocalhost%3A6006"
   ```

   **Option C — From VS Code UI:**
   - Open Command Palette: `Ctrl+Shift+P`
   - Type **Simple Browser: Show** and press Enter
   - Enter `http://localhost:6006`

## Notes

- Run the commands from the workspace root:
  `/home/e_chrisantus/Projects/multi-object-manip/verify2act`
- The TensorBoard event file is located at:
  `verify2act/output/wm_v5/tb_logs/events.out.tfevents.1777090543.csg1.3616539.0`
- If port 6006 is already in use, either kill the existing process or use a different port:
  ```bash
  tensorboard --logdir verify2act/output/wm_v5/tb_logs --port 6007
  ```
- To kill an existing TensorBoard on port 6006:
  ```bash
  fuser -k 6006/tcp
  ```
