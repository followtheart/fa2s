//===- Fa2sOps.td ------------------------------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "Fa2s/Fa2sDialect.td"

// program(start = "state") 统一容器，持有多个 state 子 op。
def Fa2s_ProgramOp : Op<Fa2s_Dialect, "program", [IsolatedFromAbove]> {
  let summary = "FA2S program root";
  let arguments = (ins OptionalAttr<StrAttr>:$start);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    (`start` `=` $start^)? attr-dict-with-keyword $body
  }];
}

// state @name { transition* }
def Fa2s_StateOp : Op<Fa2s_Dialect, "state", [IsolatedFromAbove]> {
  let summary = "State with symbol name";
  let arguments = (ins StrAttr:$name);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $name attr-dict-with-keyword $body
  }];
}

// transition {guard attrs} -> {actions} next=<symbol|none>
// 动作为顺序区域中的多个 action op。
def Fa2s_TransitionOp : Op<Fa2s_Dialect, "transition", []> {
  let summary = "Transition with guard and actions";
  let arguments = (ins
    StrAttr:$inputGuard,
    OptionalAttr<I8Attr>:$inputLit,
    StrAttr:$aguard,
    OptionalAttr<I8Attr>:$alit,
    StrAttr:$bguard,
    OptionalAttr<I8Attr>:$blit,
    OptionalAttr<StrAttr>:$next
  );
  let regions = (region AnyRegion:$actions);
  let assemblyFormat = [{
    `input` `=` $inputGuard ( `lit` $inputLit^ )? `,`
    `A` `=` $aguard ( `lit` $alit^ )? `,`
    `B` `=` $bguard ( `lit` $blit^ )?
    ( `next` `=` $next^ )?
    attr-dict-with-keyword $actions
  }];
}

// ---- Action Ops ----

def Fa2s_ReadOp : Op<Fa2s_Dialect, "read", []> {
  let summary = "Consume one input byte and set it as last";
  let assemblyFormat = [{ attr-dict }];
}

def Fa2s_WriteOp : Op<Fa2s_Dialect, "write", []> {
  let summary = "Write string or last";
  let arguments = (ins OptionalAttr<StrAttr>:$str, UnitAttr:$last);
  let assemblyFormat = [{
    ( `str` `=` $str^ )? ( `last` $last^ )? attr-dict
  }];
}

// push/pop A/B；可选 last 标志
def Fa2s_PushOp : Op<Fa2s_Dialect, "push", []> {
  let arguments = (ins StrAttr:$stack, OptionalAttr<I8Attr>:$value, UnitAttr:$useLast);
  let assemblyFormat = [{ $stack ( `value` $value^ )? ( `last` $useLast^ )? attr-dict }];
}

def Fa2s_PopOp : Op<Fa2s_Dialect, "pop", []> {
  let arguments = (ins StrAttr:$stack, UnitAttr:$toLast);
  let assemblyFormat = [{ $stack ( `to_last` $toLast^ )? attr-dict }];
}

// halt: accept/reject
def Fa2s_HaltOp : Op<Fa2s_Dialect, "halt", []> {
  let arguments = (ins StrAttr:$mode);
  let assemblyFormat = [{ $mode attr-dict }];
}