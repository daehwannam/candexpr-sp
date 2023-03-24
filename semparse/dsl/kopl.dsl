;;; -*- mode: hy -*-

;; (defaction
;;   :name 'and
;;   :act_type 'bool
;;   :param_types '[bool bool bool]
;;   :expr_dict (dict :default '(and_func @0 @1 @2)
;;                    :visual '(and @0 @1 @2))
;;   :optional_idx None
;;   :rest_idx 2)

;; (defaction
;;   :name 'or
;;   :act_type 'bool
;;   :param_types '[bool bool bool]
;;   :expr_dict (dict :default '(or_func @0 @1 @2)
;;                    :visual '(or @0 @1 @2))
;;   :optional_idx None
;;   :rest_idx 2)


(defaction
  :name 'program
  :act_type 'type
  :param_types '[bool bool bool]
  :expr_dict (dict :default '(and_func @0 @1 @2)
                   :visual '(and @0 @1 @2))
  :optional_idx None
  :rest_idx 2)
