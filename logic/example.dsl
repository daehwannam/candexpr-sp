;;; -*- mode: lisp -*-

(deftypes
  '(:object
    bool
    (:sequence
     tuple
     list)
    (:string
     rawstring
     formatstring)))

(deftypes
  '(:object
    (:immutable
     tuple)))

(defaction
  :name 'and
  :act_type 'bool
  :param_types '(bool bool bool)
  :expr_dict (mapkv :default $'(and_func @0 @1 @2)
                    :visual $'(and @0 @1 @2))
  :optional_idx None
  :rest_idx 2)

(defaction
  :name 'or
  :act_type 'bool
  :param_types '(bool bool bool)
  :expr_dict (mapkv :default $'(or_func @0 @1 @2)
                    :visual $'(or @0 @1 @2))
  :optional_idx None
  :rest_idx 2)
